let
  extras =
    import ./nix/extras.nix //
    import ./nix/gitignore.nix { inherit (pkgs) lib; };
  pkgs = extras.pinnedPkgs {
    specFile = ./nix/nixpkgs.json;
    opts = {};
  };
  python = pkgs.python37.override { packageOverrides = self: super: {
    SALib = pkgs.python37.pkgs.buildPythonPackage rec {
      pname = "SALib";
      version = "1.2";
      src = pkgs.python37.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "2ac07201d06c5c3efab9cb2284e6549c5a4e6e694bbdf1c22328af0f80afb942";
      };
      propagatedBuildInputs = [ self.matplotlib self.pandas ];
      doCheck = false;
      doInstallCheck = false;
    };
    # pymc3 requires version <0.13 which is not what's bundled in nix ATM
    joblib = pkgs.python37.pkgs.buildPythonPackage rec {
      pname = "joblib";
      version = "0.12.5";
      src = pkgs.python37.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "11cdfd38cdb71768149e1373f2509e9b4fc1ec6bc92f874cb515b25f2d69f8f4";
      };
      doCheck = false;
      doInstallCheck = false;
    };
  };};
in
(python.withPackages (ps: with ps; [ black pymc3 matplotlib numpy jupyter SALib ])).env

