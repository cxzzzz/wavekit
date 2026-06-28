module compare_xz_tb;

  logic clk;
  logic [3:0] bus;
  logic [3:0] data_0;
  logic [3:0] data_1;

  string fsdb_file;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef COMPARE_VCS
  initial begin
    if (!$value$plusargs("fsdbfile=%s", fsdb_file))
      fsdb_file = "compare_xz.fsdb";
    $fsdbDumpfile(fsdb_file);
    $fsdbDumpvars(0, compare_xz_tb, "+all");
  end
`else
  initial begin
    $dumpfile("compare_xz.vcd");
    $dumpvars(0, compare_xz_tb);
  end
`endif

  initial begin
    bus    = 4'b0000;
    data_0 = 4'b0000;
    data_1 = 4'b1111;

    @(posedge clk);
    bus    = 4'bxxxx;
    data_0 = 4'b0001;
    data_1 = 4'bzzzz;

    @(posedge clk);
    bus    = 4'bzzzz;
    data_0 = 4'bx010;
    data_1 = 4'b1010;

    @(posedge clk);
    bus    = 4'b10x1;
    data_0 = 4'b101z;
    data_1 = 4'b0x0z;

    @(posedge clk);
    bus    = 4'b1z0x;
    data_0 = 4'b1111;
    data_1 = 4'b0000;

    @(posedge clk);
    bus    = 4'b1010;
    data_0 = 4'b0000;
    data_1 = 4'b1111;

    @(posedge clk);
    $finish;
  end

endmodule
