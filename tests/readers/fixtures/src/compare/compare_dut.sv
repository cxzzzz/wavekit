module compare_dut (
  input logic clk,
  input logic rst_n
);

  logic [3:0] counter;
  logic [15:0] status;

  compare_unit unit_a (.clk(clk), .rst_n(rst_n));
  compare_unit unit_b (.clk(clk), .rst_n(rst_n));

  always @(posedge clk) begin
    if (!rst_n) begin
      counter <= 4'h0;
      status  <= 16'h0;
    end else begin
      counter <= counter + 4'h1;
      status  <= {counter, 8'hAB, counter};
    end
  end

endmodule
