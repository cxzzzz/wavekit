module simple_dut (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       valid,
  input  logic [3:0] data_i,
  output logic [3:0] data_o,
  output logic       overflow
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_o <= 4'h0;
      overflow <= 1'b0;
    end else if (valid) begin
      {overflow, data_o} <= data_i + 4'h1;
    end
  end
endmodule
