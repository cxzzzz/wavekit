module axi_lite_reg_slave (
    input wire clk,
    input wire rst_n,
    input wire arvalid,
    output wire arready,
    input wire [31:0] araddr,
    output reg rvalid,
    input wire rready,
    output reg [31:0] rdata
);
    reg busy;
    reg [31:0] latched_addr;
    reg [1:0] delay_cycles;

    assign arready = !busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            latched_addr <= 32'd0;
            delay_cycles <= 2'd0;
            rvalid <= 1'b0;
            rdata <= 32'd0;
        end else begin
            if (!busy && arvalid) begin
                busy <= 1'b1;
                latched_addr <= araddr;
                delay_cycles <= araddr[2] ? 2'd3 : 2'd2;
                rvalid <= 1'b0;
            end else if (busy && !rvalid) begin
                if (delay_cycles != 0) begin
                    delay_cycles <= delay_cycles - 1'b1;
                end else begin
                    rvalid <= 1'b1;
                    rdata <= 32'h1000_0000 | (latched_addr << 2);
                end
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
                busy <= 1'b0;
            end
        end
    end
endmodule
