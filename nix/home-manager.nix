# Home-manager module: install memkit and wire its hook into an agent
# harness's hooks directory.
#
# The module owns the hook's `~/.claude` entries (KTD2). Two things land there,
# and both have to, because the harness invokes the hook as a bare path and the
# hook resolves its wordlist beside itself:
#
#   memory-prompt-recall.py   a wrapper that execs the packaged hook with the
#                             config path BAKED IN (`--set`, not
#                             `--set-default`)
#   common-words.txt          the wordlist, at the filename the pre-extraction
#                             layout used
#
# Why a wrapper and not a symlink to the .py: the config path has to arrive
# somehow, and the two alternatives are both ruled out by KTD1. Rewriting the
# source at build time (`substituteInPlace`) forks `_VERSION` — a sha256 of the
# hook's own bytes, stamped into every soak record — on config-only changes, so
# the log splits into incomparable series. Leaving it ambient hands the
# store-roots decision to whatever repo the session is standing in, which is the
# memory-poisoning surface the design names. The wrapper sets the variable and
# execs the UNMODIFIED file, so `__file__` (and therefore `_VERSION`) is the
# store copy's, byte-stable across every consumer.
{
  lib,
  config,
  pkgs,
  ...
}:
let
  cfg = config.programs.memkit;

  generated = pkgs.writeText "memkit.json" (
    builtins.toJSON {
      schema = 1;
      inherit (cfg) roots stores;
      citations = cfg.citations;
      search_cli = cfg.searchCli;
    }
  );

  configFile = if cfg.configFile != null then cfg.configFile else generated;

  # `--set`, never `--set-default`: an ambient MEMKIT_CONFIG must not be able
  # to point the every-prompt hook at another tree's stores.
  wrapped = pkgs.symlinkJoin {
    name = "memkit-wrapped-${cfg.package.version}";
    paths = [ cfg.package ];
    nativeBuildInputs = [ pkgs.makeWrapper ];
    postBuild = ''
      for bin in $out/bin/*; do
        wrapProgram "$bin" --set MEMKIT_CONFIG ${configFile}
      done
    '';
  };

  hookEntry =
    pkgs.runCommand "memkit-hook-entry"
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
      }
      ''
        mkdir -p $out/bin
        # Any python3 will do, and saying so is the point: the hook is
        # stdlib-only with a 3.9 import floor precisely so the interpreter it
        # meets is not a correctness variable.
        makeWrapper ${pkgs.python3}/bin/python3 \
          $out/bin/memory-prompt-recall.py \
          --add-flags ${cfg.package}/share/memkit/memory_prompt_recall.py \
          --set MEMKIT_CONFIG ${configFile}
      '';
in
{
  options.programs.memkit = {
    enable = lib.mkEnableOption "the memkit memory retrieval hook and checkers";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The memkit package to install and wrap.";
    };

    hooksDir = lib.mkOption {
      type = lib.types.str;
      default = ".claude/hooks";
      description = ''
        Harness hooks directory, relative to `$HOME`. The hook entry is written
        as `memory-prompt-recall.py` inside it, which is the filename a
        pre-extraction `settings.json` already names — so adopting this module
        needs no settings edit.
      '';
    };

    configFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = lib.literalExpression ''"''${config.home.homeDirectory}/.config/nix/files/claude/memkit.json"'';
      description = ''
        A committed config file to bake into the hook. Wins over `roots` /
        `stores` / `citations`, which exist for a consumer that would rather
        keep the config in Nix than in a checked-in JSON file.

        Null AND no stores means memkit is installed and inert: no config, no
        store roots, zero pointers, exit 0.
      '';
    };

    roots = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf lib.types.anything);
      default = { };
      example = lib.literalExpression ''
        {
          canonical = {
            kind = "path";
            path = "~/.config/nix";
          };
        }
      '';
      description = ''
        Named roots, each with a resolution `kind` (`path`, `git_toplevel`, or
        `config_relative`). A `path` root is `~`-expanded when the config is
        READ, never here — redirecting `HOME` is how the test suites and a
        build sandbox point the whole tool at a fixture corpus.
      '';
    };

    stores = lib.mkOption {
      type = lib.types.listOf (lib.types.attrsOf lib.types.anything);
      default = [ ];
      example = lib.literalExpression ''
        [
          {
            id = "project";
            role = "project";
            dir = "docs/memories";
            live_root = "canonical";
          }
        ]
      '';
      description = ''
        The store list, ORDERED. Order is a contract: retrieval interleaves
        hits across store directories in this order. N stores, not a fixed
        project/personal pair.
      '';
    };

    citations = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = "Citation checking: top-level `roots`, `extra_suffixes`, `blame_base`.";
    };

    searchCli = lib.mkOption {
      type = lib.types.str;
      default = "memory-recall --search";
      description = ''
        The on-demand search recipe the hook advertises to agents when it
        truncates. It is a command string handed to a model, which is why it is
        config and never environment.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Symlinked hook entries alone would leave the recipe the hook advertises
    # command-not-found on every host (KTD2).
    home.packages = [ wrapped ];

    home.file = {
      "${cfg.hooksDir}/memory-prompt-recall.py".source = "${hookEntry}/bin/memory-prompt-recall.py";
      "${cfg.hooksDir}/common-words.txt".source = "${cfg.package}/share/memkit/common-words.txt";
    };
  };
}
