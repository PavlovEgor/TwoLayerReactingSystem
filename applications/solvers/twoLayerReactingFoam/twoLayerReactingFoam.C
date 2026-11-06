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
    twoLayerReactingFoam

Group
    grpMultiphaseSolvers

Description
    Transient solver for buoyancy-driven convection with a neutralization
    reaction A + B -> C + H2O in an immiscible two-layer system in a vertical
    Hele-Shaw cell (Bratsun et al. 2023, Phil. Trans. R. Soc. A 381: 20220178).

    Primitive variable form of eqs (2.4)-(2.10) of the paper, written for the
    gap-averaged velocity in each of the layers:

    \verbatim
        rho*(ddt(U) + 6/5*div(phi, U)) =
            - grad(p) + mu*laplacian(U) - 12*mu/h^2*U + rhok*g

        rhok = rho*(betaA*A + betaB*B + W(t)*betaC*C)

        ddt(A) + div(phi, A) = DA*laplacian(A) - k*alpha*A*B
        ddt(B) + div(phi, B) = DB*laplacian(B) - k*alpha*A*B
        ddt(C) + div(phi, C) = DC*laplacian(C) + k*alpha*A*B
    \endverbatim

    The flat undeformable interface is a set of mesh faces with no flow, no
    flux of B and C, continuous A and its flux, and continuous velocity and
    shear stress.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pimpleControl.H"
#include "twoLayerReactingSystem.H"
#include "uniformDimensionedFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Transient solver for an immiscible two-layer system"
        " with a neutralization reaction in a Hele-Shaw cell."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createTimeControls.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readTimeControls.H"
        #include "CourantNo.H"
        #include "setDeltaT.H"

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
            #include "UEqn.H"
            #include "speciesEqns.H"

            while (pimple.correct())
            {
                #include "pEqn.H"
            }
        }

        layers.writeHistory(A, B, C);

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
