module simple_tb;
  logic clk;
  logic rst_n;
  logic valid;
  logic [3:0] data_i;
  logic [3:0] data_o;
  logic overflow;
  logic [3:0] bus;
  logic [3:0] data_0;
  logic [3:0] data_1;

  simple_dut dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid(valid),
    .data_i(data_i),
    .data_o(data_o),
    .overflow(overflow)
  );

  initial begin
    string fsdb_file;

    if (!$value$plusargs("fsdbfile=%s", fsdb_file)) begin
      fsdb_file = "tests/testdata/simple.fsdb";
    end

    $fsdbDumpfile(fsdb_file);
    $fsdbDumpvars(0, simple_tb);
  end

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    valid = 1'b0;
    data_i = 4'h0;
    bus = 4'b0000;
    data_0 = 4'b0000;
    data_1 = 4'b1111;

    #12 rst_n = 1'b1;
    @(posedge clk) valid = 1'b1; data_i = 4'h1; bus = 4'bxxxx; data_0 = 4'b0001; data_1 = 4'bzzzz;
    @(posedge clk) valid = 1'b1; data_i = 4'h2; bus = 4'bzzzz; data_0 = 4'bx010; data_1 = 4'b1010;
    @(posedge clk) valid = 1'b1; data_i = 4'he; bus = 4'b10x1; data_0 = 4'b101z; data_1 = 4'b0x0z;
    @(posedge clk) valid = 1'b0; data_i = 4'h3; bus = 4'b1z0x; data_0 = 4'b1111; data_1 = 4'b0000;
    @(posedge clk) valid = 1'b1; data_i = 4'hf; bus = 4'b1010; data_0 = 4'b0000; data_1 = 4'b1111;
    @(posedge clk) valid = 1'b0; data_i = 4'h0;
    @(posedge clk) $finish;
  end
endmodule
