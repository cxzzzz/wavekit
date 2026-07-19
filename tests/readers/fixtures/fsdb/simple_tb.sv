module simple_tb;
  typedef struct packed {
    logic       valid;
    logic [2:0] data;
  } packet_t;

  typedef union packed {
    logic [3:0] raw;
    packet_t    packet;
  } packet_union_t;

  logic clk;
  logic rst_n;
  logic valid;
  logic [3:0] data_i;
  logic [3:0] data_o;
  logic overflow;
  logic [3:0] bus;
  logic [3:0] data_0;
  logic [3:0] data_1;
  logic [7:4] nonzero_vec;
  logic [0:0] zero_range_vec;
  packet_t pkt;
  packet_union_t pkt_union;
  logic [10:0][2:0] packed_arr;
  logic [10:0] unpacked_arr [2:0];
  packet_t pkt_arr [1:0];
  packet_t [1:0] pkt_packed_arr;

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
    $fsdbDumpvars(0, simple_tb, "+all");
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
    nonzero_vec = 4'hC;
    zero_range_vec = 1'b1;
    pkt = {1'b0, 3'h0};
    pkt_union.raw = 4'h0;
    packed_arr = {11{3'h0}};
    unpacked_arr[0] = 11'h000;
    unpacked_arr[1] = 11'h111;
    unpacked_arr[2] = 11'h222;
    pkt_arr[0] = {1'b0, 3'h0};
    pkt_arr[1] = {1'b1, 3'h7};
    pkt_packed_arr[0] = {1'b0, 3'h0};
    pkt_packed_arr[1] = {1'b1, 3'h7};

    #12 rst_n = 1'b1;
    @(posedge clk) valid = 1'b1; data_i = 4'h1; bus = 4'bxxxx; data_0 = 4'b0001; data_1 = 4'bzzzz; nonzero_vec = 4'hA; pkt = {1'b1, 3'h1}; pkt_union.raw = 4'h9; packed_arr = {11{3'h1}}; unpacked_arr[0] = 11'h001; unpacked_arr[1] = 11'h101; unpacked_arr[2] = 11'h201; pkt_arr[0] = {1'b1, 3'h1}; pkt_arr[1] = {1'b0, 3'h6}; pkt_packed_arr[0] = {1'b1, 3'h1}; pkt_packed_arr[1] = {1'b0, 3'h6};
    @(posedge clk) valid = 1'b1; data_i = 4'h2; bus = 4'bzzzz; data_0 = 4'bx010; data_1 = 4'b1010; nonzero_vec = 4'h5; pkt = {1'b1, 3'h2}; pkt_union.raw = 4'hA; packed_arr = {11{3'h2}}; unpacked_arr[0] = 11'h002; unpacked_arr[1] = 11'h102; unpacked_arr[2] = 11'h202; pkt_arr[0] = {1'b1, 3'h2}; pkt_arr[1] = {1'b0, 3'h5}; pkt_packed_arr[0] = {1'b1, 3'h2}; pkt_packed_arr[1] = {1'b0, 3'h5};
    @(posedge clk) valid = 1'b1; data_i = 4'he; bus = 4'b10x1; data_0 = 4'b101z; data_1 = 4'b0x0z; nonzero_vec = 4'hF; pkt = {1'b0, 3'h7}; pkt_union.raw = 4'h7; packed_arr = {11{3'h3}}; unpacked_arr[0] = 11'h003; unpacked_arr[1] = 11'h103; unpacked_arr[2] = 11'h203; pkt_arr[0] = {1'b0, 3'h3}; pkt_arr[1] = {1'b1, 3'h4}; pkt_packed_arr[0] = {1'b0, 3'h3}; pkt_packed_arr[1] = {1'b1, 3'h4};
    @(posedge clk) valid = 1'b0; data_i = 4'h3; bus = 4'b1z0x; data_0 = 4'b1111; data_1 = 4'b0000; nonzero_vec = 4'h4; pkt = {1'b1, 3'h4}; pkt_union.raw = 4'hC; packed_arr = {11{3'h4}}; unpacked_arr[0] = 11'h004; unpacked_arr[1] = 11'h104; unpacked_arr[2] = 11'h204; pkt_arr[0] = {1'b1, 3'h4}; pkt_arr[1] = {1'b0, 3'h3}; pkt_packed_arr[0] = {1'b1, 3'h4}; pkt_packed_arr[1] = {1'b0, 3'h3};
    @(posedge clk) valid = 1'b1; data_i = 4'hf; bus = 4'b1010; data_0 = 4'b0000; data_1 = 4'b1111; nonzero_vec = 4'hC; pkt = {1'b0, 3'h5}; pkt_union.raw = 4'h5; packed_arr = {11{3'h5}}; unpacked_arr[0] = 11'h005; unpacked_arr[1] = 11'h105; unpacked_arr[2] = 11'h205; pkt_arr[0] = {1'b0, 3'h5}; pkt_arr[1] = {1'b1, 3'h2}; pkt_packed_arr[0] = {1'b0, 3'h5}; pkt_packed_arr[1] = {1'b1, 3'h2};
    @(posedge clk) valid = 1'b0; data_i = 4'h0;
    @(posedge clk) $finish;
  end
endmodule
