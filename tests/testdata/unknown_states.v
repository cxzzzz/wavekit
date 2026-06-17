module tb;
  reg clk;
  reg [3:0] bus;
  reg [3:0] data_0;
  reg [3:0] data_1;

  initial begin
    $dumpfile("unknown_states.fst");
    $dumpvars(0, tb);

    clk = 1'b0;
    bus = 4'b0000;
    data_0 = 4'b0000;
    data_1 = 4'b1111;

    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'bxxxx; data_0 = 4'b0001; data_1 = 4'bzzzz;
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'bzzzz; data_0 = 4'bx010; data_1 = 4'b1010;
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b10x1; data_0 = 4'b101z; data_1 = 4'b0x0z;
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b1z0x; data_0 = 4'b1111; data_1 = 4'b0000;
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b1010; data_0 = 4'b0000; data_1 = 4'b1111;
    #5 clk = 1'b1;
    #5 clk = 1'b0;
    #1 $finish;
  end
endmodule
