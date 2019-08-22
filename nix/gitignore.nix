import ((import <nixpkgs> {}).fetchFromGitHub {
  owner = "siers";
  repo = "nix-gitignore";
  inherit (builtins.fromJSON (builtins.readFile ./nix-gitignore.json)) rev sha256;
})
