# FPGA Quickstart Guide: SystemVerilog

A beginner-friendly, strictly "read-forward" guide. Every concept, term, and syntax element is explained before or at the moment it is introduced—no looking forward required.

---

## 1. The Hardware Mental Model (No Code Yet)

Before writing a single line of SystemVerilog, you must understand what an FPGA physically is.

### Software vs. Hardware
* **Software (Temporal):** A CPU reads one instruction after another from memory and runs them sequentially on a fixed processor over time.
* **Hardware (Spatial):** An FPGA is a vast grid of uncommitted physical resources (logic gates, storage cells, and routing tracks). When you compile code for an FPGA, you are not writing instructions for a CPU to run. You are **wiring physical circuits across physical space**. Every circuit you define exists and operates simultaneously.

### The Two Physical Building Blocks
Everything inside digital hardware boils down to two fundamental elements:

```
+-------------------+      +----------------------+
| Combinational     |      | Sequential           |
| Logic (Gates/LUT) | ---> | Storage (Flip-Flop)  | ---> Next Stage
+-------------------+      +----------------------+
                                      ^
                                      | Clock Signal (Periodic Pulse)
```

1. **Combinational Logic (Gates & LUTs):**
   * **What it does:** Computes mathematical and logical decisions based *only* on the current inputs (e.g., AND, OR, addition, multiplexing).
   * **How FPGAs implement it:** Instead of millions of physical AND/OR gates, FPGAs use small programmable memory tables called **LUTs (Look-Up Tables)**. A 6-input LUT can replicate the truth table of any possible 6-input boolean function.
   * **Key physical property:** Combinational logic has **no memory**. When the inputs change, the output changes after a tiny physical delay (propagation delay).

2. **Sequential Logic (Flip-Flops / Registers):**
   * **What it does:** Stores a single bit of information (`0` or `1`).
   * **How it works:** A flip-flop (FF) listens to a rhythmic metronome called the **Clock**. On each rising edge (the transition from `0` to `1`) of the clock, the flip-flop samples its input, saves it, and holds that value steady at its output until the next clock tick.
   * **Key physical property:** Sequential logic provides **memory and synchronization**. It holds state steady so subsequent circuits can read it without it fluctuating.

### The Clock, Propagation Delay, and Timing
Electricity does not travel instantaneously. When a flip-flop outputs a new value:
1. The voltage ripples through a chain of combinational logic gates (LUTs).
2. The signals take time to settle into their final `0` or `1` state. This duration is the **propagation delay**.
3. The signal must arrive and stay completely stable at the next flip-flop slightly *before* the next clock tick arrives. This required stability window is called **Setup Time**.
4. The signal must also stay stable for a tiny instant *after* the clock tick. This is called **Hold Time**.

**The Critical Path:** The longest combinational chain between any two flip-flops determines how fast your clock can tick. If your clock ticks faster than the time it takes for signals to travel through that longest path plus the setup time, the receiving flip-flop captures garbage. This failure is called a **timing violation**.

### Two Core Hardware Invariants
1. **The Single-Driver Principle:**
   * A physical wire can only be driven by **one** source. If two gates try to drive the same wire to different voltages (one pulling to `1`, another to `0`), you create electrical contention (a short circuit). Every signal in your design must have exactly one driver.
2. **Latches vs. Flip-Flops (The "Accidental Memory" Trap):**
   * A **flip-flop** updates cleanly on a clock edge.
   * A **latch** is transparent: it opens whenever a control signal is high and freezes when low. Latches are notoriously prone to timing glitches, race conditions, and routing problems in FPGAs.
   * If you describe combinational logic but forget to specify what an output should be under every possible condition, the synthesis tool assumes you wanted to "remember" the old value—and accidentally builds a latch.

---

## 2. Representing Signals and Numbers in SystemVerilog

Now that you know hardware consists of wires and registers, let's see how SystemVerilog represents them.

### The Universal Signal Type: `logic`
In older Verilog, you had to choose between two confusing keywords: `wire` and `reg`. SystemVerilog simplifies this:
* **Use `logic` for virtually all signals.**
* A `logic` variable represents a single physical connection that can hold a digital state (`0`, `1`, `X` for unknown, `Z` for floating).

```systemverilog
logic ready;        // A 1-bit signal
logic valid;        // Another 1-bit signal
```

### Multi-Bit Buses (Vectors)
To represent numbers or multi-wire buses, declare the bit range using `[MSB:LSB]` (Most Significant Bit down to Least Significant Bit):

```systemverilog
logic [7:0]  data_byte;   // 8-bit bus (bits 7 down to 0)
logic [15:0] word_data;   // 16-bit bus
logic [31:0] memory_addr; // 32-bit bus
```
* Bit index `7` is the most significant bit (value $2^7 = 128$).
* Bit index `0` is the least significant bit (value $2^0 = 1$).

### Writing Number Literals
Hardware literals specify their exact bit-width and base using the format: `<width>'<base><value>`:
* **Binary (`'b`):** `4'b1010` (a 4-bit value equal to 10 decimal)
* **Hexadecimal (`'h`):** `8'hFF` (an 8-bit value equal to 255 decimal)
* **Decimal (`'d`):** `16'd1000` (a 16-bit value equal to 1000 decimal)
* **Fill Shortcuts:**
  * `'0` : Fills all bits of the target signal with `0` regardless of width.
  * `'1` : Fills all bits of the target signal with `1` regardless of width.

### Basic Signal Operations
You can combine, slice, and manipulate signals just like physical wires:

```systemverilog
logic [7:0] a, b;
logic [7:0] sum, bitwise_and;
logic [3:0] upper_half;
logic [15:0] combined;

// Arithmetic & Bitwise
assign sum         = a + b;     // Creates adder logic
assign bitwise_and = a & b;     // Creates 8 parallel AND gates

// Slicing: Extracting a subset of wires
assign upper_half  = a[7:4];    // Connects to bits 7, 6, 5, 4 of 'a'

// Concatenation: Bundling wires together with curly braces { }
assign combined    = {a, b};    // 16 bits total: 'a' in [15:8], 'b' in [7:0]

// Replication: Repeating bits with {multiplier{signal}}
assign combined    = {{8{a[7]}}, a}; // Sign-extends 8-bit 'a' to 16 bits
```

---

## 3. Describing Hardware: Combinational vs. Sequential

SystemVerilog uses distinct language constructs to describe the two physical building blocks introduced in Section 1.

### Combinational Logic: `assign` and `always_comb`

Combinational logic evaluates continuously. There is no clock involved.

#### Method 1: Continuous Assignment (`assign`)
Best for simple equations and data routing:

```systemverilog
logic a, b, sel;
logic y;

// Physical meaning: A 2-to-1 multiplexer (mux) gate
assign y = sel ? b : a;
```
Whenever `sel`, `a`, or `b` changes, `y` updates immediately after gate delay.

#### Method 2: Procedural Combinational (`always_comb`)
Best for complex decisions, priority encoders, and `case` statements:

```systemverilog
logic [1:0] mode;
logic [7:0] in_a, in_b;
logic [7:0] result;

always_comb begin
    // 1. ALWAYS set a default value first to prevent accidental latches!
    result = '0;

    // 2. Use blocking assignments (=) inside always_comb
    case (mode)
        2'b00:   result = in_a + in_b;
        2'b01:   result = in_a - in_b;
        2'b10:   result = in_a & in_b;
        2'b11:   result = in_a | in_b;
        default: result = '0;
    endcase
end
```

**Understanding the Latch Trap:**
Inside `always_comb`, statements describe the truth table for an output wire. If there is any branch where an output is not given a value (for example, an `if` condition without an `else`), the synthesis tool assumes the output must retain its previous value. Since pure gates cannot remember values, the tool synthesizes an unwanted physical **latch**.
* **The Rule:** Always assign a default value to every output at the very top of `always_comb`, or ensure every `if` has an `else` and every `case` has a `default`.

---

### Sequential Logic: `always_ff`

Sequential logic creates physical **flip-flops (registers)** that only update on a clock edge.

```systemverilog
logic clk;
logic rst_n;    // Reset signal (active-low: '0' means reset, '1' means run)
logic enable;
logic [7:0] d;  // Data input
logic [7:0] q;  // Registered data output

// Triggered ONLY on the rising edge of clk
always_ff @(posedge clk) begin
    if (!rst_n) begin
        q <= '0;        // Synchronous Reset: clear output to 0
    end else if (enable) begin
        q <= d;         // Capture input 'd' into register 'q'
    end
    // If neither applies, flip-flops automatically hold their value!
end
```

### The Crucial Rule: Blocking (`=`) vs. Non-Blocking (`<=`)

Notice that `always_comb` uses `=`, while `always_ff` uses `<=`. This is not optional:

1. **Blocking (`=`):**
   * Executes sequentially in code order, like software.
   * **Use exclusively inside `always_comb`.**
   * Computes intermediate steps of a combinational equation.

2. **Non-Blocking (`<=`):**
   * Models the physical behavior of flip-flops in parallel.
   * All right-hand side expressions are evaluated simultaneously using the values *before* the clock edge. Then, all left-hand side registers update together.
   * **Use exclusively inside `always_ff`.**

```systemverilog
// SWAPPING TWO REGISTERS IN HARDWARE:
// With non-blocking (<=), this works correctly in a single clock cycle!
always_ff @(posedge clk) begin
    reg_a <= reg_b; // reg_a gets old value of reg_b
    reg_b <= reg_a; // reg_b gets old value of reg_a
end
```

### Summary Comparison
| Target Hardware | SystemVerilog Construct | Assignment Operator | Latches Possible? |
| :--- | :--- | :--- | :--- |
| **Gates / Wire / Mux** | `assign` or `always_comb` | `=` (Blocking) | Yes (if incomplete) |
| **Flip-Flops / Registers** | `always_ff @(posedge clk)` | `<=` (Non-blocking) | No |

---

## 4. Packaging Hardware: Modules

A **module** is the fundamental building block of hardware hierarchy—comparable to an integrated circuit chip with physical input and output pins.

### Complete, Working Example: An 8-bit Counter
Here is how the concepts learned so far fit into a self-contained module:

```systemverilog
`default_nettype none // Disables implicit wire creation to catch typo bugs

module counter #(
    parameter int WIDTH = 8,      // Compile-time configurable width
    parameter int MAX_VAL = 255   // Compile-time maximum value
) (
    input  logic             clk,      // System clock
    input  logic             rst_n,    // Active-low synchronous reset
    input  logic             enable,   // Count enable control
    output logic [WIDTH-1:0] count,    // Counter value output
    output logic             rollover  // High for 1 cycle at max value
);

    // Internal signal to calculate the next register value
    logic [WIDTH-1:0] next_count;

    // 1. Combinational logic: calculate what the next value should be
    always_comb begin
        if (enable) begin
            if (count == MAX_VAL[WIDTH-1:0]) begin
                next_count = '0;
            end else begin
                next_count = count + 1'b1;
            end
        end else begin
            next_count = count; // Hold current value
        end
    end

    // 2. Sequential logic: register the value on each clock edge
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            count <= '0;
        end else begin
            count <= next_count;
        end
    end

    // 3. Continuous assignment: flag when rollover occurs
    assign rollover = (count == MAX_VAL[WIDTH-1:0]) && enable;

endmodule

`default_nettype wire // Restore default setting for vendor IP
```

### Connecting Modules Together (Instantiation)
To use a module inside another module, instantiate it and connect its ports by name:

```systemverilog
module top (
    input  logic clk,
    input  logic rst_n,
    output logic heart_beat
);

    logic [7:0] counter_val;
    logic       pulse;

    // Instantiate the counter module
    counter #(
        .WIDTH(8),
        .MAX_VAL(200)
    ) u_counter (
        .clk      (clk),          // .module_port(local_signal)
        .rst_n    (rst_n),
        .enable   (1'b1),         // Always enabled
        .count    (counter_val),
        .rollover (pulse)
    );

    // Toggle an LED on each rollover
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            heart_beat <= 1'b0;
        end else if (pulse) begin
            heart_beat <= ~heart_beat;
        end
    end

endmodule
```

---

## 5. Frequently Used Hardware Patterns

Now that you know how to build modules with gates and registers, here are the architectural patterns used in almost every FPGA design.

### Pattern 1: Pipelining (Beating the Critical Path)
Recall from Section 1 that long combinational calculations limit your clock speed. **Pipelining** breaks a long operation across multiple clock cycles by inserting intermediate flip-flops.

```
Unpipelined (Slow Fmax):
[Reg A] ---> [Multiply 16x16] ---> [Add 32-bit] ---> [Reg Result]  (Long delay!)

Pipelined (Fast Fmax):
[Reg A] ---> [Multiply 16x16] ---> [Stage 1 Reg] ---> [Add 32-bit] ---> [Reg Result]
```

```systemverilog
module pipelined_mac (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic [31:0] c,
    output logic [31:0] result
);

    // Pipeline registers (Stage 1)
    logic [31:0] mult_reg;
    logic [31:0] c_delay_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mult_reg    <= '0;
            c_delay_reg <= '0;
            result      <= '0;
        end else begin
            // Stage 1: Perform multiply and forward 'c'
            mult_reg    <= a * b;
            c_delay_reg <= c;

            // Stage 2: Perform addition using Stage 1 registers
            result      <= mult_reg + c_delay_reg;
        end
    end

endmodule
```

---

### Pattern 2: Finite State Machines (FSM)
An FSM is a digital controller that transitions through a predefined set of states.

The cleanest, most reliable approach is the **2-Process FSM**:
* **Process 1 (Sequential):** Updates the current state register on the clock edge.
* **Process 2 (Combinational):** Determines the next state and outputs based on current state and inputs.

```systemverilog
// 1. Declare state names using an enumerated type (enum)
typedef enum logic [1:0] {
    STATE_IDLE,
    STATE_ACTIVE,
    STATE_DONE
} state_t;

state_t current_state, next_state;
logic   start_cmd, task_complete;
logic   busy_signal;

// Process 1: State Register (always_ff)
always_ff @(posedge clk) begin
    if (!rst_n) begin
        current_state <= STATE_IDLE;
    end else begin
        current_state <= next_state;
    end
end

// Process 2: Next-State Logic & Outputs (always_comb)
always_comb begin
    // Defaults to prevent latches
    next_state  = current_state;
    busy_signal = 1'b0;

    case (current_state)
        STATE_IDLE: begin
            if (start_cmd) begin
                next_state = STATE_ACTIVE;
            end
        end

        STATE_ACTIVE: begin
            busy_signal = 1'b1;
            if (task_complete) begin
                next_state = STATE_DONE;
            end
        end

        STATE_DONE: begin
            next_state = STATE_IDLE;
        end

        default: next_state = STATE_IDLE;
    endcase
end
```

---

### Pattern 3: Block RAM (BRAM) Inference
FPGAs have dedicated embedded SRAM blocks called **BRAM**. Instead of consuming thousands of individual flip-flops for storage, you can infer BRAM by writing synchronous read/write code that matches the hardware template:

```systemverilog
module single_port_bram #(
    parameter int ADDR_WIDTH = 10,  // 1024 words (2^10)
    parameter int DATA_WIDTH = 32   // 32 bits per word
) (
    input  logic                  clk,
    input  logic                  write_en,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] write_data,
    output logic [DATA_WIDTH-1:0] read_data
);

    // Declare the memory array: [bit_width] name [depth]
    logic [DATA_WIDTH-1:0] memory [0:(1<<ADDR_WIDTH)-1];

    // Synchronous memory operation
    always_ff @(posedge clk) begin
        if (write_en) begin
            memory[addr] <= write_data;
        end
        // Synchronous read (output updates on clock edge)
        // This exact line tells the synthesis tool to use dedicated BRAM
        read_data <= memory[addr];
    end

endmodule
```

---

### Pattern 4: Hardware Replication with `generate`
When you need to stamp out multiple identical instances of a circuit in parallel, use a `generate` loop. This does not loop in time; it **duplicates hardware across space**.

```systemverilog
// Create 4 parallel pipeline stages
logic [7:0] pipe [0:3];
logic [7:0] data_in;

genvar i;
generate
    for (i = 0; i < 4; i++) begin : gen_stages
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                pipe[i] <= '0;
            end else if (i == 0) begin
                pipe[i] <= data_in;
            end else begin
                pipe[i] <= pipe[i-1]; // Shift from previous stage
            end
        end
    end
endgenerate
```

---

## 6. Structuring Larger Designs: Medium-Frequency Features

As designs scale in complexity, raw bit-vectors become error-prone. SystemVerilog provides clean abstractions for organizing data.

### Packed Structs: Bundling Wires Together
A **packed struct** groups related signals into a named structure while remaining a single, contiguous bit-vector that can be assigned, sliced, or sent across a bus:

```systemverilog
typedef struct packed {
    logic [7:0]  source_id;
    logic [7:0]  dest_id;
    logic [15:0] payload_length;
} packet_header_t;

packet_header_t header;

// Access individual fields
assign header.source_id      = 8'h01;
assign header.payload_length = 16'd64;

// Treat the whole struct as a single 32-bit vector
logic [31:0] raw_bus;
assign raw_bus = header; // Legal direct assignment
```

### Packages: Sharing Constants and Types Across Files
Instead of defining the same types in every file, define them once in a `package`:

```systemverilog
// Define in "my_project_pkg.sv"
package my_project_pkg;
    localparam int BUS_WIDTH = 64;

    typedef struct packed {
        logic [BUS_WIDTH-1:0] data;
        logic                 valid;
        logic                 ready;
    } bus_channel_t;
endpackage : my_project_pkg

// Use in any module file
module receiver
    import my_project_pkg::*; // Import all symbols from the package
(
    input  logic         clk,
    input  bus_channel_t rx_bus
);
    // Use rx_bus.data, rx_bus.valid, etc.
endmodule
```

### Interfaces: Bundling Inter-Module Cables
When two modules share 10 or 20 signals, passing each port individually is tedious. An `interface` acts as a composite cable, and `modport` enforces signal directions for each side:

```systemverilog
interface simple_bus_if (input logic clk);
    logic [31:0] data;
    logic        valid;
    logic        ready;

    // Master produces data, consumes ready
    modport master (
        input  clk, ready,
        output data, valid
    );

    // Slave consumes data, produces ready
    modport slave (
        input  clk, data, valid,
        output ready
    );
endinterface

// Usage in module:
module producer (
    simple_bus_if.master bus
);
    always_ff @(posedge bus.clk) begin
        if (bus.ready) begin
            bus.valid <= 1'b1;
            bus.data  <= 32'hDEADBEEF;
        end
    end
endmodule
```

---

## 7. Physical Realities, Edge Cases & Verification

Now that you can write complete designs, let's explore real-world physical constraints and how to verify your hardware.

### Clock Domain Crossing (CDC) & Metastability

What happens if an asynchronous signal (like a button press or data from another chip with a different clock) arrives at a flip-flop at the exact moment its clock ticks?
1. The setup or hold time is violated.
2. The flip-flop enters **metastability**: its internal voltage hovers between `0` and `1` for an unpredictable amount of time before collapsing randomly to `0` or `1`.
3. Downstream logic sees inconsistent values, causing erratic system crashes.

#### Solution for Single-Bit Signals: The 2-FF Synchronizer
Pass the asynchronous bit through two flip-flops in series. The first flip-flop absorbs the metastability, allowing the second flip-flop to output a clean, stable digital value:

```systemverilog
module bit_synchronizer (
    input  logic dest_clk,
    input  logic async_in,
    output logic sync_out
);
    // Attribute tells synthesis tools to place these FFs close together
    (* ASYNC_REG = "TRUE" *) logic stage1, stage2;

    always_ff @(posedge dest_clk) begin
        stage1   <= async_in;
        stage2   <= stage1;
    end

    assign sync_out = stage2;
endmodule
```

* **Multi-Bit Bus Warning:** You **cannot** use independent 2-FF synchronizers on a multi-bit data bus. Because wires have slightly different lengths, bits will arrive at different clock cycles, corrupting the data. Multi-bit CDC requires an **Asynchronous FIFO** or a **handshake protocol**.

---

### Signed Arithmetic Rules
By default, all numbers and operations in SystemVerilog are **unsigned**.

```systemverilog
logic signed [7:0] a, b;
logic signed [8:0] sum;

// Both operands are signed -> synthesizes signed math
assign sum = a + b;

// PITFALL: If even ONE operand in an expression is unsigned,
// SystemVerilog converts the ENTIRE expression to unsigned!
logic [7:0] unsigned_val;
assign sum = a + $signed(unsigned_val); // Must cast explicitly
```

---

### Writing a Self-Checking Testbench
Hardware cannot be debugged with `printf` breakpoints on a running chip. You verify modules in a software simulator (such as Verilator, Questa, or Vivado Simulator) using a **testbench**.

A testbench generates a simulated clock, applies input stimulus over time, and asserts that the outputs match expectations:

```systemverilog
`timescale 1ns / 1ps

module tb_counter;
    localparam time CLK_PERIOD = 10ns; // 100 MHz clock

    logic clk = 0;
    logic rst_n = 0;
    logic enable = 0;
    logic [7:0] count;
    logic rollover;

    // 1. Generate the clock signal (toggles every 5ns)
    always #(CLK_PERIOD / 2) clk = ~clk;

    // 2. Instantiate the Device Under Test (DUT)
    counter #(
        .WIDTH(8),
        .MAX_VAL(5)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .enable   (enable),
        .count    (count),
        .rollover (rollover)
    );

    // 3. Drive stimulus sequentially over simulation time
    initial begin
        // Hold reset active for 2 clock cycles
        #(CLK_PERIOD * 2);
        @(posedge clk);
        rst_n <= 1'b1; // De-assert reset

        // Enable the counter
        @(posedge clk);
        enable <= 1'b1;

        // Wait for 6 clock cycles
        repeat (6) @(posedge clk);

        // Check if output matches expectation
        if (count !== 8'd1) begin
            $fatal(1, "TEST FAILED: Expected count=1, got=%0d", count);
        end else begin
            $display("TEST PASSED: Counter rolled over correctly!");
        end

        // End simulation
        $finish;
    end

endmodule
```

---

## 8. Synthesis Safety Checklist

Before sending your SystemVerilog to synthesis, verify your code against this list:

1. [ ] **Did you use `=` for `always_comb` and `<=` for `always_ff`?**
   * Never mix them in the same block.
2. [ ] **Did you assign a default to every output in `always_comb`?**
   * Prevents accidental latches.
3. [ ] **Does every signal have exactly one driver?**
   * Prevents multi-driver contention shorts.
4. [ ] **Did you avoid gating the clock with logic gates?**
   * Never do `assign gated_clk = clk & enable;`. This creates clock skew and glitches. Use the `enable` pin of your flip-flop inside `always_ff` instead.
5. [ ] **Are you only resetting what needs a reset?**
   * Reset control logic, state machines, and valid flags.
   * Large datapath delay registers and RAM buffers do not need reset wires; omitting resets allows tools to pack them into dense hardware resources (like SRL shift-register LUTs and BRAMs).
