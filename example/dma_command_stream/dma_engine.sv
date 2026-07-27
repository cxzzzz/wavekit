module dma_engine (
    input wire clk,
    input wire rst_n,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire cmd_op,
    input wire [3:0] cmd_len,
    input wire wvalid,
    output wire wready,
    input wire [31:0] wdata,
    output reg rvalid,
    input wire rready,
    output reg [31:0] rdata,
    output reg rsp_valid,
    input wire rsp_ready,
    output reg [7:0] rsp_status
);
    localparam IDLE = 3'd0;
    localparam WRITE = 3'd1;
    localparam READ_RSP = 3'd2;
    localparam READ_DATA = 3'd3;
    localparam RESP = 3'd4;

    reg [2:0] state;
    reg [3:0] remaining_beats;
    reg [31:0] read_seed;

    assign cmd_ready = state == IDLE;
    assign wready = state == WRITE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            remaining_beats <= 4'd0;
            read_seed <= 32'h3000_0000;
            rvalid <= 1'b0;
            rdata <= 32'd0;
            rsp_valid <= 1'b0;
            rsp_status <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    rvalid <= 1'b0;
                    rsp_valid <= 1'b0;
                    if (cmd_valid && cmd_ready) begin
                        remaining_beats <= cmd_len;
                        if (cmd_op) begin
                            state <= WRITE;
                        end else begin
                            read_seed <= 32'h3000_0000 | {28'd0, cmd_len};
                            rsp_valid <= 1'b1;
                            rsp_status <= 8'd3;
                            state <= READ_RSP;
                        end
                    end
                end
                WRITE: begin
                    if (wvalid && wready) begin
                        if (remaining_beats == 4'd1) begin
                            remaining_beats <= 4'd0;
                            rsp_valid <= 1'b1;
                            rsp_status <= 8'd7;
                            state <= RESP;
                        end else begin
                            remaining_beats <= remaining_beats - 1'b1;
                        end
                    end
                end
                READ_RSP: begin
                    if (rsp_valid && rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state <= READ_DATA;
                    end
                end
                READ_DATA: begin
                    if (!rvalid && remaining_beats != 0) begin
                        rvalid <= 1'b1;
                        rdata <= read_seed;
                    end else if (rvalid && rready) begin
                        rvalid <= 1'b0;
                        read_seed <= read_seed + 32'd1;
                        if (remaining_beats == 4'd1) begin
                            remaining_beats <= 4'd0;
                            state <= IDLE;
                        end else begin
                            remaining_beats <= remaining_beats - 1'b1;
                        end
                    end
                end
                RESP: begin
                    if (rsp_valid && rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end
endmodule
