# memkit as an installed application: three console scripts, plus the hook and
# its wordlist as a LOOSE PAIR under share/memkit/.
#
# The loose pair is not redundancy. The harness invokes the hook by file path,
# the hook resolves common-words.txt beside its own __file__, and _common_words()
# degrades to an EMPTY stopword set when that file is missing rather than
# raising — a silently different retriever. Shipping the two together at one
# stable path is what the home-manager module points at.
#
# Built as a python PACKAGE and re-exported as an application by flake.nix
# (`toPythonApplication`, which only flips a passthru flag — same store path).
# An application-only build is invisible to `python3.withPackages`, and the
# suites import `memkit` rather than a source directory, precisely so a
# packaging bug cannot hide from them.
{
  lib,
  buildPythonPackage,
  hatchling,
  hatch-vcs,
  version,
}:

buildPythonPackage {
  pname = "memkit";
  inherit version;

  src = ../.;

  pyproject = true;

  build-system = [
    hatchling
    hatch-vcs
  ];

  # KTD7: a flake input materialises without `.git`, so hatch-vcs has nothing
  # to read. The flake hands the rev in here; `checks.version-matches` asserts
  # the built tool reports it, so dropping this line goes red instead of
  # silently stamping 0.0.0.
  env.HATCH_VCS_PRETEND_VERSION = version;

  postInstall = ''
    mkdir -p $out/share/memkit
    cp src/memkit/memory_prompt_recall.py src/memkit/common-words.txt \
      $out/share/memkit/
  '';

  # The wordlist has to land beside the module in site-packages too, or the
  # console script's retriever is the silently different empty-stopword one.
  pythonImportsCheck = [ "memkit" ];
  postFixup = ''
    test -f $out/share/memkit/common-words.txt
    test -f "$(dirname "$(find $out/lib -name memory_prompt_recall.py -print -quit)")/common-words.txt"
  '';

  doCheck = false; # the suites run as flake checks, against this output

  meta = {
    description = "Retrieval hook, integrity checker and eval harness for a tiered markdown memory store";
    mainProgram = "memory-recall";
    license = lib.licenses.asl20;
  };
}
