{
    volScalarField rAU("rAU", 1.0/UEqn.A());

    // Without the interface mask: used for the reconstruction only
    const surfaceScalarField rAUfFull("rAUfFull", fvc::interpolate(rAU));
    const surfaceScalarField rAUf("rAUf", rAUfFull*interfaceOpen);

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    const surfaceScalarField phig(-rAUf*ghf*fvc::snGrad(rhok)*mesh.magSf());

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        interfaceOpen
       *(
            fvc::flux(HbyA)
          + fvc::interpolate(rho*rAU)*fvc::ddtCorr(U, phi)
        )
      + phig
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvm::laplacian(rAUf, p) == fvc::div(phiHbyA)
        );

        layers.setPressureReferences(pEqn);

        pEqn.solve(p.select(pimple.finalInnerIter()));

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA - pEqn.flux();

            p.relax();

            U = HbyA + rAU*fvc::reconstruct((phig - pEqn.flux())/rAUfFull);
            U.correctBoundaryConditions();
        }
    }

    #include "continuityErrs.H"
}
