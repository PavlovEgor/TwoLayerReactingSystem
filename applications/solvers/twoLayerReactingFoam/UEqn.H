// Momentum equation of the gap-averaged flow

rhoPhiConv = (6.0/5.0)*fvc::interpolate(rho)*phi;

fvVectorMatrix UEqn
(
    fvm::ddt(rho, U)
  + fvm::div(rhoPhiConv, U)
  - fvm::laplacian(layers.muf(), U)
  + fvm::Sp(gapFriction, U)
);

layers.addInterfaceNormalStress(UEqn, U);

UEqn.relax();

if (pimple.momentumPredictor())
{
    solve
    (
        UEqn
     ==
        fvc::reconstruct
        (
            (
              - ghf*fvc::snGrad(rhok)
              - fvc::snGrad(p)
            )*mesh.magSf()*interfaceOpen
        )
    );
}
