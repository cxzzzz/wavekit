module dma_tb;
    reg clk;
    reg rst_n;
    reg cmd_valid;
    wire cmd_ready;
    reg cmd_op;
    reg [3:0] cmd_len;
    reg wvalid;
    wire wready;
    reg [31:0] wdata;
    wire rvalid;
    reg rready;
    wire [31:0] rdata;
    wire rsp_valid;
    reg rsp_ready;
    wire [7:0] rsp_status;

    dma_engine u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_op(cmd_op),
        .cmd_len(cmd_len),
        .wvalid(wvalid),
        .wready(wready),
        .wdata(wdata),
        .rvalid(rvalid),
        .rready(rready),
        .rdata(rdata),
        .rsp_valid(rsp_valid),
        .rsp_ready(rsp_ready),
        .rsp_status(rsp_status)
    );

    always #5 clk = ~clk;

    task issue_cmd(input op, input [3:0] len);
        begin
            @(posedge clk);
            cmd_op <= op;
            cmd_len <= len;
            cmd_valid <= 1'b1;
            while (!cmd_ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            cmd_valid <= 1'b0;
        end
    endtask

    task send_write_beat(input [31:0] data);
        begin
            @(posedge clk);
            wdata <= data;
            wvalid <= 1'b1;
            while (!wready) begin
                @(posedge clk);
            end
            @(posedge clk);
            wvalid <= 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        cmd_valid = 1'b0;
        cmd_op = 1'b0;
        cmd_len = 4'd0;
        wvalid = 1'b0;
        wdata = 32'd0;
        rready = 1'b1;
        rsp_ready = 1'b1;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        issue_cmd(1'b1, 4'd2);
        send_write_beat(32'd160);
        send_write_beat(32'd161);

        repeat (3) @(posedge clk);
        issue_cmd(1'b0, 4'd2);

        repeat (16) @(posedge clk);
        $finish;
    end

    initial begin
        $dumpfile("dma_tb.vcd");
        $dumpvars;
    end
endmodule
