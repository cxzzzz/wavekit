module axi_read_reorder_slave (
    input wire clk,
    input wire rst_n,
    input wire arvalid,
    output wire arready,
    input wire [3:0] arid,
    input wire [31:0] araddr,
    output reg rvalid,
    input wire rready,
    output reg [3:0] rid,
    output reg [31:0] rdata
);
    localparam IDLE = 2'd0;
    localparam DELAY = 2'd1;
    localparam RESP = 2'd2;

    reg slot0_valid;
    reg slot1_valid;
    reg [3:0] slot0_id;
    reg [3:0] slot1_id;
    reg [31:0] slot0_addr;
    reg [31:0] slot1_addr;
    reg serving_slot1;
    reg [1:0] state;
    reg [1:0] delay_cycles;

    assign arready = !(slot0_valid && slot1_valid);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            slot0_valid <= 1'b0;
            slot1_valid <= 1'b0;
            slot0_id <= 4'd0;
            slot1_id <= 4'd0;
            slot0_addr <= 32'd0;
            slot1_addr <= 32'd0;
            serving_slot1 <= 1'b0;
            delay_cycles <= 2'd0;
            rvalid <= 1'b0;
            rid <= 4'd0;
            rdata <= 32'd0;
        end else begin
            if (arvalid && arready) begin
                if (!slot0_valid) begin
                    slot0_valid <= 1'b1;
                    slot0_id <= arid;
                    slot0_addr <= araddr;
                end else begin
                    slot1_valid <= 1'b1;
                    slot1_id <= arid;
                    slot1_addr <= araddr;
                end
            end

            case (state)
                IDLE: begin
                    rvalid <= 1'b0;
                    if (slot1_valid) begin
                        serving_slot1 <= 1'b1;
                        rid <= slot1_id;
                        rdata <= 32'h2000_0000 | slot1_addr;
                        delay_cycles <= 2'd1;
                        state <= DELAY;
                    end else if (slot0_valid) begin
                        serving_slot1 <= 1'b0;
                        rid <= slot0_id;
                        rdata <= 32'h2000_0000 | slot0_addr;
                        delay_cycles <= 2'd2;
                        state <= DELAY;
                    end
                end
                DELAY: begin
                    if (delay_cycles != 0) begin
                        delay_cycles <= delay_cycles - 1'b1;
                    end else begin
                        rvalid <= 1'b1;
                        state <= RESP;
                    end
                end
                RESP: begin
                    if (rvalid && rready) begin
                        rvalid <= 1'b0;
                        if (serving_slot1) begin
                            slot1_valid <= 1'b0;
                        end else begin
                            slot0_valid <= 1'b0;
                        end
                        state <= IDLE;
                    end
                end
            endcase

            if (state == RESP && rvalid && rready) begin
                rvalid <= 1'b0;
            end
        end
    end
endmodule
