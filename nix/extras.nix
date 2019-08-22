import ((import <nixpkgs> {}).fetchFromGitHub {
  owner = "colehaus";
  repo = "nix-extras";
  inherit (builtins.fromJSON (builtins.readFile ./nix-extras.json)) rev sha256;
})
