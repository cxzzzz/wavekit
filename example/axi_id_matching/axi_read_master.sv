module axi_read_master (
    input wire clk,
    input wire rst_n,
    input wire req_valid,
    output wire req_ready,
    input wire [3:0] req_id,
    input wire [31:0] req_addr,
    output reg arvalid,
    input wire arready,
    output reg [3:0] arid,
    output reg [31:0] araddr,
    input wire rvalid,
    output wire rready,
    input wire [3:0] rid,
    input wire [31:0] rdata,
    output reg rsp_valid,
    input wire rsp_ready,
    output reg [3:0] rsp_id,
    output reg [31:0] rsp_data
);
    localparam IDLE = 2'd0;
    localparam SEND_AR = 2'd1;
    localparam HOLD_RSP = 2'd2;

    reg [1:0] state;
    reg [1:0] head;
    reg [1:0] tail;
    reg [1:0] count;
    reg [3:0] id_q [0:1];
    reg [31:0] addr_q [0:1];

    assign req_ready = count != 2'd2;
    assign rready = state != HOLD_RSP || rsp_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            head <= 2'd0;
            tail <= 2'd0;
            count <= 2'd0;
            arvalid <= 1'b0;
            arid <= 4'd0;
            araddr <= 32'd0;
            rsp_valid <= 1'b0;
            rsp_id <= 4'd0;
            rsp_data <= 32'd0;
        end else begin
            if (req_valid && req_ready) begin
                id_q[tail] <= req_id;
                addr_q[tail] <= req_addr;
                tail <= tail + 1'b1;
                count <= count + 1'b1;
            end

            if (rvalid && rready) begin
                rsp_valid <= 1'b1;
                rsp_id <= rid;
                rsp_data <= rdata;
                state <= HOLD_RSP;
            end else begin
                case (state)
                    IDLE: begin
                        rsp_valid <= 1'b0;
                        if (!arvalid && count != 0) begin
                            arid <= id_q[head];
                            araddr <= addr_q[head];
                            arvalid <= 1'b1;
                            state <= SEND_AR;
                        end else if (req_valid && req_ready) begin
                            arvalid <= 1'b1;
                            arid <= req_id;
                            araddr <= req_addr;
                            state <= SEND_AR;
                        end
                    end
                    SEND_AR: begin
                        if (arvalid && arready) begin
                            arvalid <= 1'b0;
                            head <= head + 1'b1;
                            count <= count - 1'b1;
                            state <= IDLE;
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
    end
endmodule
