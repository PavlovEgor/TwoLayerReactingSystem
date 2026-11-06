/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM, distributed under GPL-3.0-or-later.

Application
    setTwoLayerFields

Group
    grpPreProcessingUtilities

Description
    Set the initial state (2.12) of the paper from constant/twoLayerProperties:
    A = A0 in the upper layer, B = B0 in the lower layer,
    C = 0 and the velocity of a random stream function.

    \verbatim
        perturbation
        {
            amplitude   1e-3;   // Stream function amplitude, units of DA
            seed        1234;
        }
    \endverbatim

    The zero amplitude gives the quiescent base state. The mesh is expected to
    be two-dimensional in the x-y plane.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "Random.H"
#include "twoLayerReactingSystem.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote("Set the initial two-layer state");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    twoLayerReactingSystem layers(mesh);

    const dictionary pertDict(layers.subOrEmptyDict("perturbation"));
    const scalar amplitude = pertDict.getOrDefault<scalar>("amplitude", 1e-3);
    const label seed = pertDict.getOrDefault<label>("seed", 1234);

    auto readField = [&](const word& name)
    {
        return volScalarField
        (
            IOobject
            (
                name,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
    };

    volScalarField A(readField("A"));
    volScalarField B(readField("B"));
    volScalarField C(readField("C"));

    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    const scalarField& alpha = layers.alpha().primitiveField();

    A.primitiveFieldRef() =
        (1.0 - alpha)*layers.A0().value();
    A.correctBoundaryConditions();

    B.primitiveFieldRef() = alpha*layers.B0().value();
    B.correctBoundaryConditions();

    C.primitiveFieldRef() = Zero;
    C.correctBoundaryConditions();

    // Random stream function: zero on the boundaries of the domain
    volScalarField psi
    (
        IOobject("psi", runTime.timeName(), mesh),
        mesh,
        dimensionedScalar(dimViscosity, Zero)
    );

    Random rng(seed + Pstream::myProcNo());

    forAll(psi, celli)
    {
        psi[celli] =
            amplitude*layers.DA().value()*(2*rng.sample01<scalar>() - 1);
    }

    const volVectorField gradPsi(fvc::grad(psi));

    U.primitiveFieldRef().replace(vector::X, gradPsi.primitiveField().component(vector::Y));
    U.primitiveFieldRef().replace(vector::Y, -gradPsi.primitiveField().component(vector::X));
    U.primitiveFieldRef().replace(vector::Z, scalar(0));
    U.correctBoundaryConditions();

    Info<< "Lower layer cells: " << gSum(alpha) << nl
        << "Maximum perturbation velocity: "
        << gMax(mag(U.primitiveField())()) << " m/s" << nl;

    A.write();
    B.write();
    C.write();
    U.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
