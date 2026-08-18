{
  description = "Retrieval hook, integrity checker and eval harness for a tiered markdown memory store";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  # Deliberately no `inputs.<x>.follows = "nixpkgs"` anywhere: memkit is the
  # consumed leaf, and a follows here would fork every consumer off its own
  # pinned nixpkgs — the recorded "nixpkgs.follows defeats the upstream binary
  # cache" trap, one level down.

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      flake.homeManagerModules.default =
        { pkgs, lib, ... }:
        {
          imports = [ ./nix/home-manager.nix ];
          programs.memkit.package = lib.mkDefault (
            inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.default
          );
        };

      perSystem =
        {
          pkgs,
          self',
          lib,
          ...
        }:
        let
          # KTD7. No `.gitattributes` export-subst — that is what triggers the
          # recorded narHash-mismatch trap — and a `github:` input materialises
          # without `.git`, so hatch-vcs alone would stamp 0.0.0 in every nix
          # build. The rev is handed in instead.
          #
          # As a LOCAL version segment, not as the version: PEP 440 rejects a
          # bare 40-hex sha outright (measured — the build fails in hatchling's
          # metadata validation), and `<sha>-dirty` with it. `0.0.0+g<short>`
          # is the setuptools-scm shape, carries the rev, and parses. A dirty
          # tree — the state the relocated dev loop lives in — gets the
          # revless literal rather than failing evaluation.
          clean = inputs.self ? shortRev;
          version = if clean then "0.0.0+g${inputs.self.shortRev}" else "0.0.0+dev";

          memkitLib = pkgs.python3Packages.callPackage ./nix/package.nix {
            inherit version;
          };
          memkit = pkgs.python3Packages.toPythonApplication memkitLib;

          # Every check runs the suites against the INSTALLED package, not a
          # source tree: a suite that only ever ran from a checkout cannot see
          # a packaging bug, and packaging is what this repo just grew.
          pytestEnv = pkgs.python3.withPackages (ps: [
            memkitLib
            ps.pytest
          ]);

          # The home-manager module, evaluated against a stub that declares
          # only the two options it writes. Enough to prove the module
          # evaluates and to get the hook entry it would install, without
          # making home-manager an input of a repo that does not otherwise
          # need one.
          hm = lib.evalModules {
            modules = [
              (
                { lib, ... }:
                {
                  options.home.packages = lib.mkOption {
                    type = lib.types.listOf lib.types.package;
                    default = [ ];
                  };
                  options.home.file = lib.mkOption {
                    type = lib.types.attrsOf (
                      lib.types.submodule { options.source = lib.mkOption { type = lib.types.path; }; }
                    );
                    default = { };
                  };
                }
              )
              inputs.self.homeManagerModules.default
              { _module.args.pkgs = pkgs; }
              {
                programs.memkit.enable = true;
                programs.memkit.package = memkit;
                # The whole source tree, not `./tests/fixtures/memkit.json`:
                # that form copies the ONE file into the store, and the
                # fixture config resolves its stores relative to itself, so it
                # would land beside no corpus and the check would pass by
                # finding nothing.
                programs.memkit.configFile = "${inputs.self}/tests/fixtures/memkit.json";
              }
            ];
          };

          # git is not a convenience here: the citation cases build real commits
          # in a tmpdir and the checker's staleness pass shells out to
          # `git log`. Without it the suites would skip their way to green.
          suite =
            name: file:
            pkgs.runCommand "memkit-${name}" { nativeBuildInputs = [ pkgs.git ]; } ''
              cd ${inputs.self}
              ${pytestEnv}/bin/pytest -q --no-header -p no:cacheprovider ${file} 2>&1 | tee $out
            '';
        in
        {
          packages.default = memkit;
          packages.memkit = memkit;

          checks = {
            package = memkit;

            # The two suites the split moved here (U1's `tooling` disposition).
            hook-tests = suite "hook-tests" "tests/test_memory_prompt_recall.py";
            integrity-tests = suite "integrity-tests" "tests/test_memory_integrity.py";

            # KTD4: the real-corpus gate belongs to the consumer, because the
            # cases pair prompts with private memory filenames. What memkit can
            # prove is the MECHANISM, over a corpus it invented.
            fixture-eval = pkgs.runCommand "memkit-fixture-eval" { } ''
              cp -r ${inputs.self}/tests/fixtures fixtures
              chmod -R u+w fixtures
              ${memkit}/bin/memory-eval --config fixtures/memkit.json 2>&1 | tee $out
            '';

            # And that the checker runs against a config-supplied store list at
            # all — the N-store property KTD10 asks for, on a 2-store fixture.
            fixture-integrity =
              pkgs.runCommand "memkit-fixture-integrity"
                {
                  nativeBuildInputs = [ pkgs.git ];
                }
                ''
                  cp -r ${inputs.self}/tests/fixtures fixtures
                  chmod -R u+w fixtures
                  ${memkit}/bin/memory-integrity --config fixtures/memkit.json 2>&1 | tee $out
                '';
            # KTD1 + KTD2 end to end: the entry the module writes into the
            # hooks directory runs, finds its wordlist, and reads the config
            # baked into it — with nothing in the environment saying so. An
            # inert hook and a wired one both exit 0 and print nothing on a
            # prompt with no answer, so the assertion has to be a POINTER.
            home-manager-module =
              pkgs.runCommand "memkit-home-manager-module"
                {
                  entry = hm.config.home.file.".claude/hooks/memory-prompt-recall.py".source;
                  wordlist = hm.config.home.file.".claude/hooks/common-words.txt".source;
                }
                ''
                  test -f "$wordlist"
                  export HOME=$PWD
                  printf '%s' '{"session_id":"check","prompt":"sprocket backlash after the gearbox rebuild"}' \
                    | "$entry" | tee $out
                  grep -q sprocket_alignment.md $out || {
                    echo "the installed hook entry injected nothing —"
                    echo "the baked MEMKIT_CONFIG is not reaching it."
                    exit 1
                  }
                '';
          }
          // lib.optionalAttrs clean {
            # KTD7's drift check, scoped to clean builds: on a dirty tree there
            # is no rev to match and the assertion would be about the fallback
            # literal rather than about hatch-vcs.
            version-matches = pkgs.runCommand "memkit-version-matches" { } ''
              got=$(${pytestEnv}/bin/python3 -c \
                'import importlib.metadata as m; print(m.version("memkit"))')
              echo "memkit reports: $got (want ${version})"
              [ "$got" = "${version}" ] || {
                echo "ERROR: the built tool does not report the pinned rev."
                echo "HATCH_VCS_PRETEND_VERSION was probably dropped from nix/package.nix."
                exit 1
              }
              touch $out
            '';
          };

          formatter = pkgs.nixfmt-tree;
        };
    };
}
