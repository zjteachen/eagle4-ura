.PHONY: all run clean

BUILD_DIR := build
ISA_DIR := eagle4_drafter_depthn_1_0/isa
COMPONENT_DIR := eagle4_drafter_depthn_1_0/src
ISA_EXEC := $(BUILD_DIR)/isa_eagle4
TOP_MODULE := eagle4_isa_top
TB_DIR := $(ISA_DIR)/tb

# make builds the simulator; make run also executes it.
all: $(ISA_EXEC)

run: $(ISA_EXEC)
	$(abspath $(ISA_EXEC))

$(ISA_EXEC): $(wildcard $(ISA_DIR)/*.sv) $(wildcard $(COMPONENT_DIR)/*.sv) \
		$(wildcard $(ISA_DIR)/*.svh) $(wildcard $(COMPONENT_DIR)/*.svh) Makefile
	mkdir -p $(BUILD_DIR)
	verilator --binary --timing --trace --Wno-fatal --top-module $(TOP_MODULE) \
		-I$(ISA_DIR) -I$(COMPONENT_DIR) \
		$(wildcard $(ISA_DIR)/*.sv) $(wildcard $(COMPONENT_DIR)/*.sv) \
		-Mdir $(BUILD_DIR)/obj_isa -o $(abspath $@)

# Testbenches live in $(TB_DIR) as tb_<name>.sv with a matching tb_<name>
# top module. Build just the bench plus what it instantiates; -y/+libext
# resolves the design modules from the ISA and component source trees.
$(BUILD_DIR)/tb_%: $(TB_DIR)/tb_%.sv \
		$(wildcard $(ISA_DIR)/*.sv) $(wildcard $(ISA_DIR)/*.svh) \
		$(wildcard $(TB_DIR)/*.svh) \
		$(wildcard $(COMPONENT_DIR)/*.sv) $(wildcard $(COMPONENT_DIR)/*.svh) Makefile
	mkdir -p $(BUILD_DIR)
	verilator --binary --timing --trace --Wno-fatal --top-module tb_$* \
		-y $(ISA_DIR) -y $(COMPONENT_DIR) +libext+.sv \
		-I$(ISA_DIR) -I$(COMPONENT_DIR) \
		$(TB_DIR)/tb_$*.sv \
		-Mdir $(BUILD_DIR)/obj_tb_$* -o $(abspath $@)

clean:
	rm -rf $(BUILD_DIR)
