Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading species fields\n" << endl;
volScalarField A
(
    IOobject
    (
        "A",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField B
(
    IOobject
    (
        "B",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField C
(
    IOobject
    (
        "C",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading twoLayerProperties\n" << endl;
twoLayerReactingSystem layers(mesh);
layers.printSimilarity();

const volScalarField& rho = layers.rho();
const surfaceScalarField& interfaceOpen = layers.open();

// Face flux, closed on the interface
surfaceScalarField phi
(
    IOobject
    (
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(U)*interfaceOpen
);

// Momentum flux with the 6/5 factor of the gap-averaged convective term
surfaceScalarField rhoPhiConv
(
    "rhoPhiConv",
    (6.0/5.0)*fvc::interpolate(rho)*phi
);

// Damping by the walls of the gap, 12*mu/h^2
const volScalarField gapFriction
(
    "gapFriction",
    12.0*layers.mu()/sqr(layers.hGap())
);

// Diffusivities of the species confined to the lower layer
const surfaceScalarField DBf("DBf", layers.DB()*interfaceOpen);
const surfaceScalarField DCf("DCf", layers.DC()*interfaceOpen);

#include "readGravitationalAcceleration.H"

const volScalarField gh("gh", g & mesh.C());
const surfaceScalarField ghf("ghf", g & mesh.Cf());

layers.correctWaterRelease(A, B, C);

volScalarField rhok(layers.rhok(A, B, C));

mesh.setFluxRequired(p.name());
