module axi_lite_master (
    input wire clk,
    input wire rst_n,
    input wire req_valid,
    output wire req_ready,
    input wire [31:0] req_addr,
    output reg arvalid,
    input wire arready,
    output reg [31:0] araddr,
    input wire rvalid,
    output wire rready,
    input wire [31:0] rdata,
    output reg rsp_valid,
    input wire rsp_ready,
    output reg [31:0] rsp_data
);
    localparam IDLE = 2'd0;
    localparam SEND_AR = 2'd1;
    localparam WAIT_R = 2'd2;
    localparam HOLD_RSP = 2'd3;

    reg [1:0] state;

    assign req_ready = state == IDLE;
    assign rready = (state == WAIT_R) && (!rsp_valid || rsp_ready);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            arvalid <= 1'b0;
            araddr <= 32'd0;
            rsp_valid <= 1'b0;
            rsp_data <= 32'd0;
        end else begin
            case (state)
                IDLE: begin
                    rsp_valid <= 1'b0;
                    if (req_valid) begin
                        araddr <= req_addr;
                        arvalid <= 1'b1;
                        state <= SEND_AR;
                    end
                end
                SEND_AR: begin
                    if (arvalid && arready) begin
                        arvalid <= 1'b0;
                        state <= WAIT_R;
                    end
                end
                WAIT_R: begin
                    if (rvalid) begin
                        rsp_data <= rdata;
                        rsp_valid <= 1'b1;
                        state <= HOLD_RSP;
                    end
                end
                HOLD_RSP: begin
                    if (rsp_valid && rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end
endmodule
