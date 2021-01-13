#include "distributed-ml-mpi.h"
#include "ns3/string.h"
#include <dlfcn.h>


namespace ns3 {

// void MpiHelper::Enable(int argc, std::string arg){
// 	char *args;
// 	strcpy(args, arg.c_str());;
// 	char** argv = &args;
// 	std::cout << "args: " << argv[0] << std::endl;
// 	MpiInterface::Enable (&argc, &argv);
// }




void MpiHelper::Enable(std::vector<char*> argv){
	std::cout << "argv: " << argv[0] << std::endl;
	int argc = (int)argv.size();
	char** carg;
	for(auto i=0; i<int(argv.size()); i++){
		*(carg+i) = argv[i];
	}
	std::cout << "carg: " << *carg << std::endl;

	dlopen("libmpi.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
	MpiInterface::Enable (&argc, &carg);
}



}