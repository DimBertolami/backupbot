
export interface ChartData {
    timestamp: string;
    chartPaths: {
        candlestick?: string;
        indicators?: string;
        indicators_html?: string; // HTML version of the indicators chart using Plotly
        signals?: string;
        price_3d_html?: string; // 3D price surface visualization
        signals_3d_html?: string; // 3D trading signals visualization
    };
}
