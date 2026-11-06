// Reaction-diffusion-advection of the acid, the base and the salt.
// The reaction proceeds in the lower layer only.
{
    const volScalarField kLower("kLower", layers.k()*layers.alpha());

    fvScalarMatrix AEqn
    (
        fvm::ddt(A)
      + fvm::div(phi, A)
      - fvm::laplacian(layers.DA(), A)
      + fvm::Sp(kLower*B, A)
    );
    AEqn.relax();
    AEqn.solve();

    fvScalarMatrix BEqn
    (
        fvm::ddt(B)
      + fvm::div(phi, B)
      - fvm::laplacian(DBf, B)
      + fvm::Sp(kLower*A, B)
    );
    BEqn.relax();
    BEqn.solve();

    fvScalarMatrix CEqn
    (
        fvm::ddt(C)
      + fvm::div(phi, C)
      - fvm::laplacian(DCf, C)
     ==
        kLower*A*B
    );
    CEqn.relax();
    CEqn.solve();

    layers.correctWaterRelease(A, B, C);

    rhok = layers.rhok(A, B, C);
}
