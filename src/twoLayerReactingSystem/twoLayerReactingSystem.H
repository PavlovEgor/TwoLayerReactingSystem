/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM, distributed under GPL-3.0-or-later.

Class
    Foam::twoLayerReactingSystem

Description
    Physical model of an immiscible two-layer system with a neutralization
    reaction A + B -> C + H2O in a vertical Hele-Shaw cell:

    Bratsun D, Mizev A, Utochkin V, Nekrasov S, Shmyrova A. 2023
    Nonlinear development of convective patterns driven by a neutralization
    reaction in immiscible two-layer systems.
    Phil. Trans. R. Soc. A 381: 20220178.

    The undeformable interface is a set of internal faces between the two
    layers. It is impermeable for the flow, the base and the salt, and is
    transparent for the acid and for the shear stress.

    The class owns the layer indicator and the layer properties, the interface
    face mask and the water release function W(t) of eqs (2.13)-(2.14):

    \verbatim
        W = 1/(1 + dVw/Vrz),   dVw/Vrz = Vm*int(C dA)/(L*omega)

        omega = int(z*R dA)/int(R dA),   R = k*A*B
    \endverbatim

    with z measured from the interface along gravity into the lower layer.

Usage
    Dictionary constant/twoLayerProperties:
    \verbatim
        hGap            1.2e-3;     // Gap width                 [m]
        rhoLower        1000;       // Lower layer density       [kg/m3]
        rhoUpper        810.8;      // Upper layer density       [kg/m3]
        nuLower         1e-6;       // Lower layer viscosity     [m2/s]
        nuUpper         4.81e-6;    // Upper layer viscosity     [m2/s]
        DA              3.344e-9;   // Acid diffusivity          [m2/s]
        DB              2.140e-9;   // Base diffusivity          [m2/s]
        DC              1.605e-9;   // Salt diffusivity          [m2/s]
        betaA           1.063e-5;   // Expansion coefficients    [m3/mol]
        betaB           2.594e-5;
        betaC           2.424e-5;
        A0              1000;       // Initial acid              [mol/m3]
        B0              400;        // Initial base              [mol/m3]
        Da              1000;       // Damkoehler number, k*A0*h^2/DA
        molarVolume     4.5e-5;     // mu_H2O/rho_H2O + mu_NaCl/rho_NaCl
        waterRelease    true;       // Use W(t), else W = 1
        interfacePoint  (0 0 0);    // A point on the interface
    \endverbatim

SourceFiles
    twoLayerReactingSystem.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_twoLayerReactingSystem_H
#define Foam_twoLayerReactingSystem_H

#include "IOdictionary.H"
#include "dimensionedScalar.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvMatricesFwd.H"
#include "OFstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class fvMesh;

/*---------------------------------------------------------------------------*\
                   Class twoLayerReactingSystem Declaration
\*---------------------------------------------------------------------------*/

class twoLayerReactingSystem
:
    public IOdictionary
{
    // Private Data

        //- Reference to the mesh
        const fvMesh& mesh_;

        //- Gravity direction, unit vector
        vector gHat_;

        //- Gap width
        dimensionedScalar hGap_;

        //- Layer densities and kinematic viscosities
        dimensionedScalar rhoLower_;
        dimensionedScalar rhoUpper_;
        dimensionedScalar nuLower_;
        dimensionedScalar nuUpper_;

        //- Species diffusivities
        dimensionedScalar DA_;
        dimensionedScalar DB_;
        dimensionedScalar DC_;

        //- Solutal expansion coefficients
        dimensionedScalar betaA_;
        dimensionedScalar betaB_;
        dimensionedScalar betaC_;

        //- Initial concentrations
        dimensionedScalar A0_;
        dimensionedScalar B0_;

        //- Damkoehler number
        scalar Da_;

        //- Reaction rate constant, k = Da*DA/(A0*h^2)
        dimensionedScalar k_;

        //- Volume released per mole of salt
        dimensionedScalar molarVolume_;

        //- Account for the water release
        Switch waterRelease_;

        //- A point on the interface
        vector interfacePoint_;

        //- Lower layer indicator (1 in the lower layer, 0 in the upper one)
        volScalarField alpha_;

        //- Layer density
        volScalarField rho_;

        //- Layer dynamic viscosity
        volScalarField mu_;

        //- Face mask: 0 on the interface, 1 elsewhere
        surfaceScalarField open_;

        //- Face dynamic viscosity (harmonic mean on the interface)
        surfaceScalarField muf_;

        //- Interface area
        scalar interfaceArea_;

        //- Pressure reference cells of the two layers (-1: not on this rank)
        label refCellLower_;
        label refCellUpper_;

        //- Dilution function W
        scalar W_;

        //- Reaction zone width (dimensional)
        scalar omega_;

        //- Water volume increment per unit gap width, Vm*int(C dA)
        scalar dVw_;

        //- Output file (master only)
        autoPtr<OFstream> historyPtr_;


    // Private Member Functions

        //- Create the face mask from the layer indicator
        void calcInterface();

        //- Find a reference cell in the region selected by the mask
        label findRefCell(const bool lowerLayer) const;

        //- Add the normal stress correction for a cell on the interface
        void addNormalStress
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
        ) const;


public:

    //- Runtime type information
    ClassName("twoLayerReactingSystem");


    // Constructors

        //- Construct from mesh
        explicit twoLayerReactingSystem(const fvMesh& mesh);


    // Member Functions

    // Access

        const volScalarField& alpha() const noexcept { return alpha_; }
        const volScalarField& rho() const noexcept { return rho_; }
        const volScalarField& mu() const noexcept { return mu_; }
        const surfaceScalarField& open() const noexcept { return open_; }
        const surfaceScalarField& muf() const noexcept { return muf_; }

        const dimensionedScalar& hGap() const noexcept { return hGap_; }
        const dimensionedScalar& DA() const noexcept { return DA_; }
        const dimensionedScalar& DB() const noexcept { return DB_; }
        const dimensionedScalar& DC() const noexcept { return DC_; }
        const dimensionedScalar& k() const noexcept { return k_; }
        const dimensionedScalar& A0() const noexcept { return A0_; }
        const dimensionedScalar& B0() const noexcept { return B0_; }

        //- The current value of the dilution function
        scalar W() const noexcept { return W_; }

        //- Unit vector in the gravity direction
        const vector& gHat() const noexcept { return gHat_; }

        //- A point on the interface
        const vector& interfacePoint() const noexcept
        {
            return interfacePoint_;
        }


    // Evaluation

        //- Excess density of the Boussinesq buoyancy term, eq (2.8)
        tmp<volScalarField> rhok
        (
            const volScalarField& A,
            const volScalarField& B,
            const volScalarField& C
        ) const;

        //- Update W(t) from the current fields, eqs (2.13)-(2.14)
        void correctWaterRelease
        (
            const volScalarField& A,
            const volScalarField& B,
            const volScalarField& C
        );

        //- Replace the viscous flux of the normal velocity on the interface
        //- by the one for the impermeable interface in each layer
        void addInterfaceNormalStress
        (
            fvMatrix<vector>& UEqn,
            const volVectorField& U
        ) const;

        //- Fix the level of the pressure in both layers
        void setPressureReferences(fvMatrix<scalar>& pEqn) const;

        //- Print the dimensionless groups of Table 1
        void printSimilarity() const;

        //- Write a line of integral quantities
        void writeHistory
        (
            const volScalarField& A,
            const volScalarField& B,
            const volScalarField& C
        );

        //- IOdictionary content is not re-read at run time
        virtual bool read() { return true; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
