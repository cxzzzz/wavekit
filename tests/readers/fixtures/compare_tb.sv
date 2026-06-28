module compare_tb;

  logic clk;
  logic rst_n;

  compare_dut dut (.clk(clk), .rst_n(rst_n));

  string fsdb_file;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef COMPARE_VCS
  initial begin
    if (!$value$plusargs("fsdbfile=%s", fsdb_file))
      fsdb_file = "compare.fsdb";
    $fsdbDumpfile(fsdb_file);
    $fsdbDumpvars(0, compare_tb, "+all");
  end
`else
  initial begin
    $dumpfile("compare.vcd");
    $dumpvars(0, compare_tb);
  end
`endif

  initial begin
    rst_n = 1'b0;

    dut.counter = 4'h0;
    dut.status  = 16'h0;

    dut.unit_a.data             = 8'h00;
    dut.unit_a.nonzero_data     = 4'hC;
    dut.unit_a.zero_range       = 1'b1;
    dut.unit_a.pkt              = 7'h00;
    dut.unit_a.u.raw            = 6'h00;
    dut.unit_a.unpacked_arr[0]  = 8'h00;
    dut.unit_a.unpacked_arr[1]  = 8'h11;
    dut.unit_a.unpacked_arr[2]  = 8'h22;
    dut.unit_a.packed_arr       = 12'h000;
    dut.unit_a.struct_arr[0]    = 7'h00;
    dut.unit_a.struct_arr[1]    = 7'h7F;
    dut.unit_a.packed_struct_arr[0] = 7'h00;
    dut.unit_a.packed_struct_arr[1] = 7'h7F;
    dut.unit_a.nested_pkt       = {7'h00, 4'h0};
    dut.unit_a.nested_arr_struct = {4'h0, 1'b0};
    dut.unit_a.unpacked_s.a     = 4'h0;
    dut.unit_a.unpacked_s.b     = 8'h00;
    dut.unit_a.bus          = 4'b0000;
    dut.unit_a.data_0       = 4'b0000;
    dut.unit_a.data_1       = 4'b1111;
    dut.unit_a.gen_blk[0].gen_sig = 4'h0;
    dut.unit_a.gen_blk[1].gen_sig = 4'h1;
    dut.unit_a.gen_blk[2].gen_sig = 4'h2;

    dut.unit_b.data             = 8'hFF;
    dut.unit_b.nonzero_data     = 4'h3;
    dut.unit_b.zero_range       = 1'b0;
    dut.unit_b.pkt              = 7'h5A;
    dut.unit_b.u.raw            = 6'h2A;
    dut.unit_b.unpacked_arr[0]  = 8'hFF;
    dut.unit_b.unpacked_arr[1]  = 8'hEE;
    dut.unit_b.unpacked_arr[2]  = 8'hDD;
    dut.unit_b.packed_arr       = 12'hFFF;
    dut.unit_b.struct_arr[0]    = 7'h15;
    dut.unit_b.struct_arr[1]    = 7'h3C;
    dut.unit_b.packed_struct_arr[0] = 7'h15;
    dut.unit_b.packed_struct_arr[1] = 7'h3C;
    dut.unit_b.nested_pkt       = {7'h15, 4'hA};
    dut.unit_b.nested_arr_struct = {4'h3, 1'b1};
    dut.unit_b.unpacked_s.a     = 4'hA;
    dut.unit_b.unpacked_s.b     = 8'hBC;
    dut.unit_b.bus          = 4'b0000;
    dut.unit_b.data_0       = 4'b0000;
    dut.unit_b.data_1       = 4'b1111;
    dut.unit_b.gen_blk[0].gen_sig = 4'h3;
    dut.unit_b.gen_blk[1].gen_sig = 4'h4;
    dut.unit_b.gen_blk[2].gen_sig = 4'h5;

    #12 rst_n = 1'b1;

    @(posedge clk);
    dut.unit_a.data = 8'hAA;
    dut.unit_a.bus = 4'b1010;
    dut.unit_a.data_0 = 4'b0001;
    dut.unit_a.data_1 = 4'b1110;
    dut.unit_a.pkt = 7'h5A;
    dut.unit_a.packed_arr = 12'hABC;
    dut.unit_a.struct_arr[0] = 7'h5A;
    dut.unit_a.gen_blk[0].gen_sig = 4'hA;

    dut.unit_b.data = 8'h55;
    dut.unit_b.bus = 4'b0101;
    dut.unit_b.data_0 = 4'b0010;
    dut.unit_b.data_1 = 4'b1100;
    dut.unit_b.pkt = 7'h3C;
    dut.unit_b.packed_arr = 12'h123;
    dut.unit_b.struct_arr[0] = 7'h3C;

    @(posedge clk);
    dut.unit_a.data = 8'h12;
    dut.unit_a.bus = 4'b1100;
    dut.unit_a.data_0 = 4'b1010;
    dut.unit_a.data_1 = 4'b0101;
    dut.unit_a.nonzero_data = 4'h5;
    dut.unit_a.u.raw = 6'h2A;

    dut.unit_b.data = 8'h34;
    dut.unit_b.bus = 4'b0011;
    dut.unit_b.data_0 = 4'b1100;
    dut.unit_b.data_1 = 4'b1010;
    dut.unit_b.nonzero_data = 4'hA;
    dut.unit_b.u.raw = 6'h15;

    @(posedge clk);
    dut.unit_a.data = 8'hFF;
    dut.unit_a.bus = 4'b1111;
    dut.unit_a.data_0 = 4'b0101;
    dut.unit_a.data_1 = 4'b1010;
    dut.unit_a.pkt = 7'h7F;
    dut.unit_a.gen_blk[1].gen_sig = 4'hB;

    dut.unit_b.data = 8'h00;
    dut.unit_b.bus = 4'b0000;
    dut.unit_b.data_0 = 4'b1111;
    dut.unit_b.data_1 = 4'b0000;
    dut.unit_b.pkt = 7'h00;
    dut.unit_b.gen_blk[1].gen_sig = 4'hC;

    @(posedge clk);
    dut.unit_a.data = 8'h00;
    dut.unit_a.bus = 4'b1x0x;
    dut.unit_a.data_0 = 4'b1111;
    dut.unit_a.data_1 = 4'b0000;
    dut.unit_a.packed_arr = 12'h000;

    dut.unit_b.data = 8'hAA;
    dut.unit_b.bus = 4'b1010;
    dut.unit_b.data_0 = 4'b0000;
    dut.unit_b.data_1 = 4'b1111;
    dut.unit_b.packed_arr = 12'hFFF;

    @(posedge clk);
    dut.unit_a.data = 8'h42;
    dut.unit_a.bus = 4'b1010;
    dut.unit_a.data_0 = 4'b0000;
    dut.unit_a.data_1 = 4'b1111;

    dut.unit_b.data = 8'h84;
    dut.unit_b.bus = 4'b0101;
    dut.unit_b.data_0 = 4'b0001;
    dut.unit_b.data_1 = 4'b1110;

    repeat (15) @(posedge clk);

    $finish;
  end

endmodule
