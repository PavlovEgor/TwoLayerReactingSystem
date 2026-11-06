/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM, distributed under GPL-3.0-or-later.

\*---------------------------------------------------------------------------*/

#include "twoLayerReactingSystem.H"
#include "fvMesh.H"
#include "fvMatrices.H"
#include "fvc.H"
#include "linear.H"
#include "uniformDimensionedFields.H"
#include "zeroGradientFvPatchFields.H"
#include "functionObject.H"
#include "PstreamReduceOps.H"
#include "Switch.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(twoLayerReactingSystem, 0);
}


// * * * * * * * * * * * * * * * * Local helpers * * * * * * * * * * * * * * //

namespace
{

Foam::IOobject propertiesIO(const Foam::fvMesh& mesh)
{
    return Foam::IOobject
    (
        "twoLayerProperties",
        mesh.time().constant(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE
    );
}

Foam::vector gravityDirection(const Foam::fvMesh& mesh)
{
    const Foam::uniformDimensionedVectorField g
    (
        Foam::IOobject
        (
            "g",
            mesh.time().constant(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        )
    );

    return g.value()/Foam::mag(g.value());
}

} // End anonymous namespace


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

void Foam::twoLayerReactingSystem::calcInterface()
{
    // A face between the layers has a value strictly between 0 and 1
    const surfaceScalarField alphaf(linearInterpolate(alpha_));

    const scalar small01 = 1e-6;

    auto isInterface = [=](const scalar a)
    {
        return a > small01 && a < 1 - small01;
    };

    scalar area = 0;

    surfaceScalarField::Internal& openI = open_.ref();
    const surfaceScalarField::Internal& alphafI = alphaf.internalField();

    forAll(openI, facei)
    {
        openI[facei] = isInterface(alphafI[facei]) ? 0 : 1;

        if (!openI[facei])
        {
            area += mesh_.magSf()[facei];
        }
    }

    auto& openBf = open_.boundaryFieldRef();

    forAll(openBf, patchi)
    {
        const fvsPatchScalarField& alphap = alphaf.boundaryField()[patchi];
        fvsPatchScalarField& openp = openBf[patchi];

        forAll(openp, i)
        {
            openp[i] = 1;

            if (alphap.coupled() && isInterface(alphap[i]))
            {
                openp[i] = 0;

                // Counted by both of the ranks sharing the face
                area += 0.5*mesh_.magSf().boundaryField()[patchi][i];
            }
        }
    }

    interfaceArea_ = returnReduce(area, sumOp<scalar>());

    if (interfaceArea_ <= 0)
    {
        FatalErrorInFunction
            << "No faces between the layers: check field " << alpha_.name()
            << exit(FatalError);
    }

    Info<< "Interface area: " << interfaceArea_ << " m2" << endl;
}


Foam::label Foam::twoLayerReactingSystem::findRefCell
(
    const bool lowerLayer
) const
{
    label refCell = -1;

    forAll(alpha_, celli)
    {
        if ((alpha_[celli] > 0.5) == lowerLayer)
        {
            refCell = celli;
            break;
        }
    }

    // The lowest rank with a cell of the layer owns the reference
    const label owner = returnReduce
    (
        refCell >= 0 ? Pstream::myProcNo() : Pstream::nProcs(),
        minOp<label>()
    );

    if (owner == Pstream::nProcs())
    {
        FatalErrorInFunction
            << "No cells in the " << (lowerLayer ? "lower" : "upper")
            << " layer" << exit(FatalError);
    }

    return owner == Pstream::myProcNo() ? refCell : -1;
}


void Foam::twoLayerReactingSystem::addNormalStress
(
    fvMatrix<vector>& UEqn,
    const label celli,
    const vector& n,
    const scalar area,
    const scalar deltaCell,
    const scalar deltaTotal,
    const scalar faceMu,
    const vector& Ucell,
    const vector& Uother
) const
{
    // Wanted force along n: -muCell*area*(U & n)/deltaCell, treated implicitly.
    // A matrix diagonal acts on every component, so the tangential part
    // is returned to the source.
    const scalar coeff = mu_[celli]*area/deltaCell;

    // Force already present in the matrix from the implicit Laplacian
    const scalar present = faceMu*area*((Uother - Ucell) & n)/deltaTotal;

    UEqn.diag()[celli] += coeff;
    UEqn.source()[celli] += coeff*(Ucell - (Ucell & n)*n) - present*n;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::twoLayerReactingSystem::twoLayerReactingSystem(const fvMesh& mesh)
:
    IOdictionary(propertiesIO(mesh)),
    mesh_(mesh),
    gHat_(gravityDirection(mesh)),
    hGap_("hGap", dimLength, *this),
    rhoLower_("rhoLower", dimDensity, *this),
    rhoUpper_("rhoUpper", dimDensity, *this),
    nuLower_("nuLower", dimViscosity, *this),
    nuUpper_("nuUpper", dimViscosity, *this),
    DA_("DA", dimViscosity, *this),
    DB_("DB", dimViscosity, *this),
    DC_("DC", dimViscosity, *this),
    betaA_("betaA", dimVolume/dimMoles, *this),
    betaB_("betaB", dimVolume/dimMoles, *this),
    betaC_("betaC", dimVolume/dimMoles, *this),
    A0_("A0", dimMoles/dimVolume, *this),
    B0_("B0", dimMoles/dimVolume, *this),
    Da_(get<scalar>("Da")),
    k_("k", Da_*DA_/(A0_*sqr(hGap_))),
    molarVolume_("molarVolume", dimVolume/dimMoles, *this),
    waterRelease_(get<Switch>("waterRelease")),
    interfacePoint_(getOrDefault<vector>("interfacePoint", Zero)),
    alpha_
    (
        IOobject
        (
            "alpha.lower",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, Zero),
        fvPatchFieldBase::zeroGradientType()
    ),
    rho_
    (
        IOobject("rho", mesh.time().timeName(), mesh),
        mesh,
        dimensionedScalar(dimDensity, Zero),
        fvPatchFieldBase::zeroGradientType()
    ),
    mu_
    (
        IOobject("mu", mesh.time().timeName(), mesh),
        mesh,
        dimensionedScalar(dimDynamicViscosity, Zero),
        fvPatchFieldBase::zeroGradientType()
    ),
    open_
    (
        IOobject("interfaceOpen", mesh.time().timeName(), mesh),
        mesh,
        dimensionedScalar(dimless, scalar(1))
    ),
    muf_
    (
        IOobject("muf", mesh.time().timeName(), mesh),
        mesh,
        dimensionedScalar(dimDynamicViscosity, Zero)
    ),
    interfaceArea_(0),
    refCellLower_(-1),
    refCellUpper_(-1),
    W_(1),
    omega_(0),
    dVw_(0)
{
    // The interface plane splits the mesh into the layers
    forAll(alpha_, celli)
    {
        alpha_[celli] =
            ((mesh.C()[celli] - interfacePoint_) & gHat_) > 0 ? 1 : 0;
    }
    alpha_.correctBoundaryConditions();

    rho_ = alpha_*rhoLower_ + (scalar(1) - alpha_)*rhoUpper_;
    rho_.correctBoundaryConditions();

    mu_ = alpha_*rhoLower_*nuLower_ + (scalar(1) - alpha_)*rhoUpper_*nuUpper_;
    mu_.correctBoundaryConditions();

    // Harmonic mean: continuity of the velocity and of the shear stress on
    // the interface. It is the viscosity itself inside of a layer.
    muf_ = scalar(1)/linearInterpolate(scalar(1)/mu_);

    calcInterface();

    refCellLower_ = findRefCell(true);
    refCellUpper_ = findRefCell(false);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField> Foam::twoLayerReactingSystem::rhok
(
    const volScalarField& A,
    const volScalarField& B,
    const volScalarField& C
) const
{
    return tmp<volScalarField>::New
    (
        "rhok",
        rho_*(betaA_*A + betaB_*B + W_*betaC_*C)
    );
}


void Foam::twoLayerReactingSystem::correctWaterRelease
(
    const volScalarField& A,
    const volScalarField& B,
    const volScalarField& C
)
{
    const scalarField& V = mesh_.V();
    const vectorField& cc = mesh_.C();

    // Layer thickness of the two-dimensional mesh from the interface area
    // cancels out: both the integrals below are per unit of the mesh depth.
    scalar intC = 0;
    scalar intR = 0;
    scalar intzR = 0;

    forAll(V, celli)
    {
        const scalar R = alpha_[celli]*A[celli]*B[celli];
        const scalar z = (cc[celli] - interfacePoint_) & gHat_;

        intC += alpha_[celli]*C[celli]*V[celli];
        intR += R*V[celli];
        intzR += z*R*V[celli];
    }

    reduce(intC, sumOp<scalar>());
    reduce(intR, sumOp<scalar>());
    reduce(intzR, sumOp<scalar>());

    // Water volume per unit of the interface length and depth
    dVw_ = molarVolume_.value()*intC/interfaceArea_;

    omega_ = intR > VSMALL ? intzR/intR : 0;

    const scalar ratio = omega_ > VSMALL ? dVw_/omega_ : 0;

    W_ = waterRelease_ ? 1/(1 + ratio) : 1;
}


void Foam::twoLayerReactingSystem::addInterfaceNormalStress
(
    fvMatrix<vector>& UEqn,
    const volVectorField& U
) const
{
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const vectorField& cc = mesh_.C();
    const surfaceVectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceScalarField& magSf = mesh_.magSf();

    forAll(owner, facei)
    {
        if (open_[facei] > 0.5) continue;

        const label own = owner[facei];
        const label nei = neighbour[facei];

        const vector n(Sf[facei]/magSf[facei]);
        const scalar dOwn = (Cf[facei] - cc[own]) & n;
        const scalar dNei = (cc[nei] - Cf[facei]) & n;

        addNormalStress
        (
            UEqn, own, n, magSf[facei], dOwn, dOwn + dNei,
            muf_[facei], U[own], U[nei]
        );
        addNormalStress
        (
            UEqn, nei, n, magSf[facei], dNei, dOwn + dNei,
            muf_[facei], U[nei], U[own]
        );
    }

    // Interface faces on the processor boundaries
    forAll(mesh_.boundary(), patchi)
    {
        const fvPatch& pp = mesh_.boundary()[patchi];

        if (!pp.coupled()) continue;

        const fvsPatchScalarField& openp = open_.boundaryField()[patchi];

        const vectorField Unei(U.boundaryField()[patchi].patchNeighbourField());
        const vectorField cNei(pp.Cn() + pp.delta());
        const labelUList& faceCells = pp.faceCells();

        forAll(openp, i)
        {
            if (openp[i] > 0.5) continue;

            const label celli = faceCells[i];
            const vector n(Sf.boundaryField()[patchi][i]/magSf.boundaryField()[patchi][i]);
            const vector& fc = Cf.boundaryField()[patchi][i];
            const scalar dCell = (fc - cc[celli]) & n;
            const scalar dNei = (cNei[i] - fc) & n;

            addNormalStress
            (
                UEqn, celli, n, magSf.boundaryField()[patchi][i],
                dCell, dCell + dNei,
                muf_.boundaryField()[patchi][i], U[celli], Unei[i]
            );
        }
    }
}


void Foam::twoLayerReactingSystem::setPressureReferences
(
    fvMatrix<scalar>& pEqn
) const
{
    // No flow crosses the interface: the two pressure systems are decoupled
    // and each of them needs its own level.
    pEqn.setReference(refCellLower_, scalar(0), true);
    pEqn.setReference(refCellUpper_, scalar(0), true);
}


void Foam::twoLayerReactingSystem::printSimilarity() const
{
    const dimensionedScalar g
    (
        dimAcceleration,
        mag
        (
            uniformDimensionedVectorField
            (
                IOobject
                (
                    "g",
                    mesh_.time().constant(),
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            ).value()
        )
    );

    const scalar tau = (sqr(hGap_)/DA_).value();

    Info<< nl << "Dimensionless groups (Table 1 of the paper):" << nl
        << "    Sc     = " << (nuLower_/DA_).value() << nl
        << "    Da     = " << (k_*A0_*sqr(hGap_)/DA_).value() << nl
        << "    RA     = "
        << (g*betaA_*A0_*pow3(hGap_)/(DA_*nuLower_)).value() << nl
        << "    RB     = " << (betaB_/betaA_).value() << nl
        << "    RC     = " << (betaC_/betaA_).value() << nl
        << "    DB     = " << (DB_/DA_).value() << nl
        << "    DC     = " << (DC_/DA_).value() << nl
        << "    nu     = " << (nuUpper_/nuLower_).value() << nl
        << "    eta    = "
        << ((rhoUpper_*nuUpper_)/(rhoLower_*nuLower_)).value() << nl
        << "    gamma  = " << (B0_/A0_).value() << nl
        << "    t unit = h^2/DA = " << tau << " s" << nl
        << "    waterRelease = " << waterRelease_ << nl << endl;
}


void Foam::twoLayerReactingSystem::writeHistory
(
    const volScalarField& A,
    const volScalarField& B,
    const volScalarField& C
)
{
    const scalarField& V = mesh_.V();
    const volVectorField& U = mesh_.lookupObject<volVectorField>("U");

    // Per unit of the mesh depth
    const scalar depth =
        (mesh_.bounds().max() - mesh_.bounds().min()) & vector(0, 0, 1);

    scalar intA = 0, intB = 0, intC = 0, intAlower = 0, intR = 0;
    scalar maxUl = 0, maxUu = 0;

    forAll(V, celli)
    {
        const scalar a = alpha_[celli];

        intA += A[celli]*V[celli];
        intAlower += a*A[celli]*V[celli];
        intB += B[celli]*V[celli];
        intC += C[celli]*V[celli];
        intR += a*k_.value()*A[celli]*B[celli]*V[celli];

        (a > 0.5 ? maxUl : maxUu) =
            max(a > 0.5 ? maxUl : maxUu, mag(U[celli]));
    }

    reduce(intA, sumOp<scalar>());
    reduce(intAlower, sumOp<scalar>());
    reduce(intB, sumOp<scalar>());
    reduce(intC, sumOp<scalar>());
    reduce(intR, sumOp<scalar>());
    reduce(maxUl, maxOp<scalar>());
    reduce(maxUu, maxOp<scalar>());

    const scalar ratio = omega_ > VSMALL ? dVw_/omega_ : 0;

    if (Pstream::master())
    {
        if (!historyPtr_)
        {
            const fileName dir
            (
                mesh_.time().globalPath()/functionObject::outputPrefix
               /"twoLayerReactingSystem"
               /mesh_.time().timeName(mesh_.time().startTime().value())
            );
            mkDir(dir);

            historyPtr_.reset(new OFstream(dir/"history.dat"));
            historyPtr_()
                << "# time[s] W omega[m] dVw/Vrz intA[mol/m] intB[mol/m]"
                << " intC[mol/m] intAlower[mol/m] intR[mol/m/s]"
                << " maxUlower[m/s] maxUupper[m/s]" << nl;
        }

        historyPtr_()
            << mesh_.time().value() << token::SPACE
            << W_ << token::SPACE
            << omega_ << token::SPACE
            << ratio << token::SPACE
            << intA/depth << token::SPACE
            << intB/depth << token::SPACE
            << intC/depth << token::SPACE
            << intAlower/depth << token::SPACE
            << intR/depth << token::SPACE
            << maxUl << token::SPACE
            << maxUu << nl;
        historyPtr_().flush();
    }
}


// ************************************************************************* //
