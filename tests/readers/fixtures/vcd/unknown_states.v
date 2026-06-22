module tb;
  typedef struct packed {
    logic       valid;
    logic [2:0] data;
  } packet_t;

  reg clk;
  reg [3:0] bus;
  reg [3:0] data_0;
  reg [3:0] data_1;
  packet_t pkt;
  logic [10:0][2:0] packed_arr;
  logic [10:0] unpacked_arr [2:0];
  packet_t pkt_arr [1:0];
  packet_t [1:0] pkt_packed_arr;

  initial begin
`ifdef FST_TRACE
    $dumpfile("unknown_states.fst");
`else
    $dumpfile("unknown_states.vcd");
`endif
    $dumpvars(0, tb);

    clk = 1'b0;
    bus = 4'b0000;
    data_0 = 4'b0000;
    data_1 = 4'b1111;
    pkt = {1'b0, 3'h0};
    packed_arr = {11{3'h0}};
    unpacked_arr[0] = 11'h000;
    unpacked_arr[1] = 11'h111;
    unpacked_arr[2] = 11'h222;
    pkt_arr[0] = {1'b0, 3'h0};
    pkt_arr[1] = {1'b1, 3'h7};
    pkt_packed_arr[0] = {1'b0, 3'h0};
    pkt_packed_arr[1] = {1'b1, 3'h7};

    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'bxxxx; data_0 = 4'b0001; data_1 = 4'bzzzz; pkt = {1'b1, 3'h1}; packed_arr = {11{3'h1}}; unpacked_arr[0] = 11'h001; unpacked_arr[1] = 11'h101; unpacked_arr[2] = 11'h201; pkt_arr[0] = {1'b1, 3'h1}; pkt_arr[1] = {1'b0, 3'h6}; pkt_packed_arr[0] = {1'b1, 3'h1}; pkt_packed_arr[1] = {1'b0, 3'h6};
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'bzzzz; data_0 = 4'bx010; data_1 = 4'b1010; pkt = {1'b1, 3'h2}; packed_arr = {11{3'h2}}; unpacked_arr[0] = 11'h002; unpacked_arr[1] = 11'h102; unpacked_arr[2] = 11'h202; pkt_arr[0] = {1'b1, 3'h2}; pkt_arr[1] = {1'b0, 3'h5}; pkt_packed_arr[0] = {1'b1, 3'h2}; pkt_packed_arr[1] = {1'b0, 3'h5};
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b10x1; data_0 = 4'b101z; data_1 = 4'b0x0z; pkt = {1'b0, 3'h7}; packed_arr = {11{3'h3}}; unpacked_arr[0] = 11'h003; unpacked_arr[1] = 11'h103; unpacked_arr[2] = 11'h203; pkt_arr[0] = {1'b0, 3'h3}; pkt_arr[1] = {1'b1, 3'h4}; pkt_packed_arr[0] = {1'b0, 3'h3}; pkt_packed_arr[1] = {1'b1, 3'h4};
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b1z0x; data_0 = 4'b1111; data_1 = 4'b0000; pkt = {1'b1, 3'h4}; packed_arr = {11{3'h4}}; unpacked_arr[0] = 11'h004; unpacked_arr[1] = 11'h104; unpacked_arr[2] = 11'h204; pkt_arr[0] = {1'b1, 3'h4}; pkt_arr[1] = {1'b0, 3'h3}; pkt_packed_arr[0] = {1'b1, 3'h4}; pkt_packed_arr[1] = {1'b0, 3'h3};
    #5 clk = 1'b1;
    #5 clk = 1'b0; bus = 4'b1010; data_0 = 4'b0000; data_1 = 4'b1111; pkt = {1'b0, 3'h5}; packed_arr = {11{3'h5}}; unpacked_arr[0] = 11'h005; unpacked_arr[1] = 11'h105; unpacked_arr[2] = 11'h205; pkt_arr[0] = {1'b0, 3'h5}; pkt_arr[1] = {1'b1, 3'h2}; pkt_packed_arr[0] = {1'b0, 3'h5}; pkt_packed_arr[1] = {1'b1, 3'h2};
    #5 clk = 1'b1;
    #5 clk = 1'b0;
    #1 $finish;
  end
endmodule
