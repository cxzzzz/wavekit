module axi_read_reorder_tb;
    reg clk;
    reg rst_n;
    reg req_valid;
    wire req_ready;
    reg [3:0] req_id;
    reg [31:0] req_addr;
    wire arvalid;
    wire arready;
    wire [3:0] arid;
    wire [31:0] araddr;
    wire rvalid;
    wire rready;
    wire [3:0] rid;
    wire [31:0] rdata;
    wire rsp_valid;
    reg rsp_ready;
    wire [3:0] rsp_id;
    wire [31:0] rsp_data;

    axi_read_master u_master (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .req_id(req_id),
        .req_addr(req_addr),
        .arvalid(arvalid),
        .arready(arready),
        .arid(arid),
        .araddr(araddr),
        .rvalid(rvalid),
        .rready(rready),
        .rid(rid),
        .rdata(rdata),
        .rsp_valid(rsp_valid),
        .rsp_ready(rsp_ready),
        .rsp_id(rsp_id),
        .rsp_data(rsp_data)
    );

    axi_read_reorder_slave u_slave (
        .clk(clk),
        .rst_n(rst_n),
        .arvalid(arvalid),
        .arready(arready),
        .arid(arid),
        .araddr(araddr),
        .rvalid(rvalid),
        .rready(rready),
        .rid(rid),
        .rdata(rdata)
    );

    always #5 clk = ~clk;

    task issue_read(input [3:0] id, input [31:0] addr);
        begin
            @(posedge clk);
            req_id <= id;
            req_addr <= addr;
            req_valid <= 1'b1;
            while (!req_ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            req_valid <= 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        req_valid = 1'b0;
        req_id = 4'd0;
        req_addr = 32'd0;
        rsp_ready = 1'b1;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        issue_read(4'd0, 32'h10);
        issue_read(4'd1, 32'h20);

        repeat (14) @(posedge clk);
        $finish;
    end

    initial begin
        $dumpfile("axi_read_reorder_tb.vcd");
        $dumpvars;
    end
endmodule
