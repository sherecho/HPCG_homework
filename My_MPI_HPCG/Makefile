# -*- Makefile -*-

arch = MY_MPI
setup_file = setup/Make.$(arch)

include $(setup_file)


HPCG_DEPS = src/CG.o \
	    src/CG_ref.o \
	    src/TestCG.o \
	    src/ComputeResidual.o \
	    src/ExchangeHalo.o \
	    src/GenerateGeometry.o \
	    src/GenerateProblem.o \
	    src/GenerateProblem_ref.o \
	    src/CheckProblem.o \
	    src/MixedBaseCounter.o \
	    src/OptimizeProblem.o \
	    src/ReadHpcgDat.o \
	    src/ReportResults.o \
	    src/SetupHalo.o \
	    src/SetupHalo_ref.o \
	    src/TestSymmetry.o \
	    src/TestNorms.o \
	    src/WriteProblem.o \
	    src/YAML_Doc.o \
	    src/YAML_Element.o \
	    src/ComputeDotProduct.o \
	    src/ComputeDotProduct_ref.o \
	    src/mytimer.o \
	    src/ComputeOptimalShapeXYZ.o \
	    src/ComputeSPMV.o \
	    src/ComputeSPMV_ref.o \
		src/ComputeSPMV_my.o \
	    src/ComputeSYMGS.o \
	    src/ComputeSYMGS_ref.o \
	    src/ComputeWAXPBY.o \
	    src/ComputeWAXPBY_ref.o \
	    src/ComputeMG_ref.o \
	    src/ComputeMG.o \
	    src/ComputeProlongation_ref.o \
	    src/ComputeRestriction_ref.o \
	    src/CheckAspectRatio.o \
	    src/OutputFile.o \
	    src/GenerateCoarseProblem.o \
	    src/init.o \
	    src/finalize.o

# These header files are included in many source files, so we recompile every file if one or more of these header is modified.
PRIMARY_HEADERS = /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/Geometry.hpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/SparseMatrix.hpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/Vector.hpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CGData.hpp \
                  /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/MGData.hpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/hpcg.hpp

all: bin/xhpcg

bin/xhpcg: src/main.o $(HPCG_DEPS)
	$(LINKER) $(LINKFLAGS) src/main.o $(HPCG_DEPS) $(HPCG_LIBS) -o bin/xhpcg

clean:
	rm -f src/*.o bin/xhpcg

.PHONY: all clean

src/main.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/main.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/CG.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CG.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/CG_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CG_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/TestCG.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestCG.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestCG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeResidual.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeResidual.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeResidual.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ExchangeHalo.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ExchangeHalo.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ExchangeHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/GenerateGeometry.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateGeometry.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateGeometry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/GenerateProblem.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateProblem.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/GenerateProblem_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateProblem_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateProblem_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/CheckProblem.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CheckProblem.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CheckProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/MixedBaseCounter.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/MixedBaseCounter.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/MixedBaseCounter.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/OptimizeProblem.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/OptimizeProblem.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/OptimizeProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ReadHpcgDat.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ReadHpcgDat.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ReadHpcgDat.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ReportResults.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ReportResults.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ReportResults.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/SetupHalo.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/SetupHalo.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/SetupHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/SetupHalo_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/SetupHalo_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/SetupHalo_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/TestSymmetry.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestSymmetry.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestSymmetry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/TestNorms.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestNorms.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/TestNorms.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/WriteProblem.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/WriteProblem.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/WriteProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/YAML_Doc.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/YAML_Doc.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/YAML_Doc.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/YAML_Element.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/YAML_Element.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/YAML_Element.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeDotProduct.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeDotProduct.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeDotProduct.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeDotProduct_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeDotProduct_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeDotProduct_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/finalize.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/finalize.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/init.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/init.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/mytimer.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/mytimer.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/mytimer.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeOptimalShapeXYZ.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeOptimalShapeXYZ.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeOptimalShapeXYZ.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeSPMV.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeSPMV_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@
src/ComputeSPMV_my.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV_my.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSPMV_my.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeSYMGS.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSYMGS.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSYMGS.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeSYMGS_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSYMGS_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeSYMGS_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeWAXPBY.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeWAXPBY.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeWAXPBY.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeWAXPBY_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeWAXPBY_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeWAXPBY_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeMG_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeMG_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeMG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeMG.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeMG.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeMG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeProlongation_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeProlongation_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeProlongation_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/ComputeRestriction_ref.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeRestriction_ref.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/ComputeRestriction_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/GenerateCoarseProblem.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateCoarseProblem.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/GenerateCoarseProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/CheckAspectRatio.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CheckAspectRatio.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/CheckAspectRatio.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

src/OutputFile.o: /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/OutputFile.cpp /public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src/OutputFile.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/public1/home/sc81071/CAS_PP2024/wangzhe/hpcg-master/src $< -o $@

