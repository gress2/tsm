#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "ok\n";

  auto test_tensor = torch::tensor({static_cast<double>(200), static_cast<double>(30), static_cast<double>(10), static_cast<double>(10)});

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(test_tensor);

  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output << std::endl;
}
