{
  buildPythonPackage,
  poetry-core,
  breakpointHook,
  python3Packages,
  gguf-py
}@inputs:

buildPythonPackage {
  pname = "llama-scripts";
  src = ../../.;
  version = "0.0.0";
  pyproject = true;
  nativeBuildInputs = [ poetry-core ];
  projectDir = ../../.;
  propagatedBuildInputs = with python3Packages; [
    numpy
    sentencepiece
    transformers
    protobuf
    torchWithoutCuda
    gguf-py
  ];
}
