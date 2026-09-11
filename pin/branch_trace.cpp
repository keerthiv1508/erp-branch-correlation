#include "pin.H"
#include <iostream>
#include <fstream>

std::ofstream outfile;

VOID RecordBranch(ADDRINT ip, BOOL taken) {
    outfile << "0x" << std::hex << ip
            << " " << (taken ? "taken" : "not_taken")
            << std::endl;
}

VOID Instruction(INS ins, VOID *v) {
    if (INS_IsBranch(ins) && INS_HasFallThrough(ins)) {
        INS_InsertCall(
            ins,
            IPOINT_BEFORE,
            (AFUNPTR)RecordBranch,
            IARG_INST_PTR,
            IARG_BRANCH_TAKEN,
            IARG_END
        );
    }
}

VOID ImageLoad(IMG img, VOID *v) {
    if (!IMG_IsMainExecutable(img)) return; // skipping the shared libraries so only the compiled function is traced.

    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)) {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)) {
            RTN_Open(rtn);
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins)) {
                Instruction(ins, v);
            }
            RTN_Close(rtn);
        }
    }
}

VOID Fini(INT32 code, VOID *v) {
    outfile.close();
}

int main(int argc, char *argv[]) {
    PIN_InitSymbols();
    PIN_Init(argc, argv);
    outfile.open("branch_trace.out");
    IMG_AddInstrumentFunction(ImageLoad, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
