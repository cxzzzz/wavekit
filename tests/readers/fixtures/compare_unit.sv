module compare_unit (
  input logic clk,
  input logic rst_n
);

  typedef struct packed {
    logic [1:0] a;
    logic       b;
    logic [2:0] c;
  } pkt_t;

  typedef union packed {
    logic [5:0] raw;
    pkt_t       as_pkt;
  } union_t;

  logic [7:0] data;
  logic [7:4] nonzero_data;
  logic [0:0] zero_range;

  pkt_t pkt;
  union_t u;

  logic [7:0] unpacked_arr [0:2];
  logic [2:0][3:0] packed_arr;
  pkt_t struct_arr [0:1];
  pkt_t [1:0] packed_struct_arr;

  struct packed {
    pkt_t       sub_pkt;
    logic [3:0] extra;
  } nested_pkt;

  struct packed {
    logic [1:0][1:0] a;
    logic            b;
  } nested_arr_struct;

  struct {
    logic [3:0] a;
    logic [7:0] b;
  } unpacked_s;

  logic [3:0] bus;
  logic [3:0] data_0;
  logic [3:0] data_1;

  genvar i;
  generate
    for (i = 0; i < 3; i++) begin : gen_blk
      logic [3:0] gen_sig;
    end
  endgenerate

endmodule
