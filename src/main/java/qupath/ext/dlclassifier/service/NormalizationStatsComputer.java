package qupath.ext.dlclassifier.service;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.regions.RegionRequest;

/**
 * Computes image-level normalization statistics for the DL pixel classifier.
 * <p>
 * Normalization stats are used to standardize pixel values before inference,
 * ensuring consistent model predictions across different images. Three sources
 * of stats are supported (in priority order):
 * <ol>
 *   <li>Training dataset stats from model metadata (most accurate)</li>
 *   <li>Image-level stats via tile sampling (~1-3s one-time cost)</li>
 *   <li>Per-tile normalization (fallback, can cause tile boundary artifacts)</li>
 * </ol>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class NormalizationStatsComputer {

    private static final Logger logger = LoggerFactory.getLogger(NormalizationStatsComputer.class);

    /** Number of tiles sampled across the image for normalization stats. */
    private static final int STATS_GRID_SIZE = 4;
    /** Maximum pixel samples per channel for stats computation. */
    private static final int STATS_TARGET_SAMPLES = 100_000;

    private NormalizationStatsComputer() {
        // Static utility class
    }

    /**
     * Computes or retrieves precomputed normalization statistics and returns a
     * ChannelConfiguration with the stats attached.
     * <p>
     * Priority order:
     * <ol>
     *   <li>Training dataset stats from model metadata (Phase 2, most accurate)</li>
     *   <li>Image-level stats via tile sampling (Phase 1, ~1-3s one-time cost)</li>
     *   <li>Per-tile normalization (fallback, can cause tile boundary artifacts)</li>
     * </ol>
     * <p>
     * FIXED_RANGE normalization always uses the user-specified values directly.
     *
     * @param server        the image server to sample from
     * @param metadata      classifier metadata (for training stats and input dimensions)
     * @param channelConfig channel configuration
     * @param contextScale  multi-scale context factor (1 = no context)
     * @param downsample    downsample factor for tile reading
     * @return channelConfig with precomputed stats, or original if not applicable
     */
    public static ChannelConfiguration compute(
            ImageServer<BufferedImage> server,
            ClassifierMetadata metadata,
            ChannelConfiguration channelConfig,
            int contextScale,
            double downsample) {
        // FIXED_RANGE uses user-specified values, no sampling needed
        if (channelConfig.getNormalizationStrategy() == ChannelConfiguration.NormalizationStrategy.FIXED_RANGE) {
            logger.info("FIXED_RANGE normalization: skipping image-level stats");
            return channelConfig;
        }

        // Priority 1: Use training dataset stats from model metadata
        if (metadata.hasNormalizationStats()) {
            List<Map<String, Double>> stats = new ArrayList<>(metadata.getNormalizationStats());
            int expectedChannels = metadata.getEffectiveInputChannels();
            // For multi-scale context: if training stats only cover detail channels
            // (older models), duplicate them for context channels as an approximation.
            // Newer models trained after the effective_channels fix already have
            // stats for all channels (detail + context).
            if (contextScale > 1 && stats.size() < expectedChannels) {
                logger.info("Expanding {} training stats to {} for context channels", stats.size(), expectedChannels);
                List<Map<String, Double>> baseStats = new ArrayList<>(stats);
                while (stats.size() < expectedChannels) {
                    // Duplicate from base stats cyclically
                    stats.add(baseStats.get(stats.size() % baseStats.size()));
                }
            }
            logger.info(
                    "Using training dataset normalization stats from model metadata " + "({} channels)", stats.size());
            return channelConfig.withPrecomputedStats(stats);
        }

        // Priority 2: Compute image-level stats via sampling
        try {
            List<Map<String, Double>> stats =
                    computeImageNormalizationStats(server, metadata, channelConfig, downsample);
            if (stats != null && !stats.isEmpty()) {
                // For multi-scale context: duplicate detail stats for context channels.
                // Context tiles come from the same image at a different scale, so the
                // pixel value distribution is similar (same staining, same dynamic range).
                if (contextScale > 1) {
                    int expectedChannels = metadata.getEffectiveInputChannels();
                    List<Map<String, Double>> expanded = new ArrayList<>(stats);
                    List<Map<String, Double>> baseStats = new ArrayList<>(stats);
                    while (expanded.size() < expectedChannels) {
                        expanded.add(baseStats.get(expanded.size() % baseStats.size()));
                    }
                    stats = expanded;
                }
                logger.info(
                        "Computed image-level normalization stats from {} sample tiles " + "({} channels)",
                        STATS_GRID_SIZE * STATS_GRID_SIZE,
                        stats.size());
                return channelConfig.withPrecomputedStats(stats);
            }
        } catch (Exception e) {
            logger.warn(
                    "Failed to compute image-level normalization stats, "
                            + "falling back to per-tile normalization: {}",
                    e.getMessage());
        }

        // Priority 3: Fall back to original config (per-tile normalization)
        return channelConfig;
    }

    /**
     * Sample image-level per-channel pixel statistics independently of
     * the normalization pipeline. Returns the base (detail-channel)
     * stats only, so the result is directly comparable to
     * {@code ClassifierMetadata.getNormalizationStats()} regardless of
     * whether multi-scale context is enabled.
     * <p>
     * Used by the out-of-distribution checker: when training stats are
     * present, {@link #compute(ImageServer, ClassifierMetadata,
     * ChannelConfiguration, int, double)} short-circuits to those and
     * never samples the image, so this method provides the image-side
     * half of the comparison.
     *
     * @return list of per-channel stat maps, or null on failure
     */
    public static List<Map<String, Double>> sampleImageStats(
            ImageServer<BufferedImage> server,
            ClassifierMetadata metadata,
            ChannelConfiguration channelConfig,
            double downsample) {
        try {
            return computeImageNormalizationStats(server, metadata, channelConfig, downsample);
        } catch (IOException e) {
            logger.warn("Failed to sample image stats for OOD check: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Samples the image to compute per-channel normalization statistics.
     * <p>
     * Reads tiles from a grid of sample locations across the image,
     * collects pixel values using reservoir sampling, and computes
     * aggregate statistics for each channel.
     *
     * @param server        the image server to sample from
     * @param metadata      classifier metadata for input dimensions
     * @param channelConfig channel configuration for selected channels
     * @param downsample    downsample factor for tile reading
     * @return list of per-channel stat maps, or null on failure
     */
    private static List<Map<String, Double>> computeImageNormalizationStats(
            ImageServer<BufferedImage> server,
            ClassifierMetadata metadata,
            ChannelConfiguration channelConfig,
            double downsample)
            throws IOException {
        int imgWidth = server.getWidth();
        int imgHeight = server.getHeight();

        // Tile dimensions at full resolution (pre-downsample)
        int tileW = (int) (metadata.getInputWidth() * downsample);
        int tileH = (int) (metadata.getInputHeight() * downsample);
        tileW = Math.min(tileW, imgWidth);
        tileH = Math.min(tileH, imgHeight);

        // Determine number of channels we'll be extracting
        List<Integer> selectedChannels = channelConfig.getSelectedChannels();
        int numChannels = selectedChannels.isEmpty() ? server.nChannels() : selectedChannels.size();

        // Collect pixel samples per channel
        List<List<Float>> channelSamples = new ArrayList<>();
        for (int c = 0; c < numChannels; c++) {
            channelSamples.add(new ArrayList<>());
        }

        // Compute grid step sizes (full resolution coordinates)
        int gridSize = Math.min(STATS_GRID_SIZE, Math.max(1, Math.min(imgWidth / tileW, imgHeight / tileH)));
        int stepX = (imgWidth - tileW) / Math.max(1, gridSize - 1);
        int stepY = (imgHeight - tileH) / Math.max(1, gridSize - 1);

        // Estimate subsample rate to keep total around TARGET_SAMPLES per channel
        int pixelsPerTile = metadata.getInputWidth() * metadata.getInputHeight();
        int totalEstimatedPixels = pixelsPerTile * gridSize * gridSize;
        int subsampleRate = Math.max(1, totalEstimatedPixels / STATS_TARGET_SAMPLES);

        int sampledTiles = 0;
        for (int gy = 0; gy < gridSize; gy++) {
            for (int gx = 0; gx < gridSize; gx++) {
                int x = Math.min(gx * stepX, imgWidth - tileW);
                int y = Math.min(gy * stepY, imgHeight - tileH);

                RegionRequest req = RegionRequest.createInstance(server.getPath(), downsample, x, y, tileW, tileH);
                BufferedImage tile = server.readRegion(req);
                if (tile == null) continue;

                Raster raster = tile.getRaster();
                int bands = raster.getNumBands();
                int dataType = raster.getDataBuffer().getDataType();
                boolean isUint8 = (dataType == DataBuffer.TYPE_BYTE);

                // Determine which bands to sample
                int w = tile.getWidth();
                int h = tile.getHeight();
                int pixelIndex = 0;

                for (int py = 0; py < h; py++) {
                    for (int px = 0; px < w; px++) {
                        pixelIndex++;
                        if (pixelIndex % subsampleRate != 0) continue;

                        for (int c = 0; c < numChannels; c++) {
                            int band = selectedChannels.isEmpty() ? c : selectedChannels.get(c);
                            if (band >= bands) continue;

                            // Keep raw pixel values (uint8 in [0, 255], float as-is).
                            // Must match the scale of values sent to Python for
                            // normalization: training metadata stats are in raw range.
                            float val;
                            if (isUint8) {
                                val = (float) (raster.getSample(px, py, band) & 0xFF);
                            } else {
                                val = raster.getSampleFloat(px, py, band);
                            }
                            channelSamples.get(c).add(val);
                        }
                    }
                }
                sampledTiles++;
            }
        }

        if (sampledTiles == 0) {
            logger.warn("No tiles could be sampled for normalization stats");
            return null;
        }

        // Compute per-channel statistics from collected samples
        List<Map<String, Double>> channelStats = new ArrayList<>();
        for (int c = 0; c < numChannels; c++) {
            List<Float> samples = channelSamples.get(c);
            if (samples.isEmpty()) {
                // Default stats for empty channel
                Map<String, Double> stats = new HashMap<>();
                stats.put("p1", 0.0);
                stats.put("p99", 1.0);
                stats.put("min", 0.0);
                stats.put("max", 1.0);
                stats.put("mean", 0.5);
                stats.put("std", 0.25);
                channelStats.add(stats);
                continue;
            }

            // Sort for percentile computation
            float[] arr = new float[samples.size()];
            for (int i = 0; i < arr.length; i++) arr[i] = samples.get(i);
            Arrays.sort(arr);

            int n = arr.length;
            double p1 = arr[Math.max(0, (int) (n * 0.01))];
            double p99 = arr[Math.min(n - 1, (int) (n * 0.99))];
            double min = arr[0];
            double max = arr[n - 1];

            // Compute mean and std
            double sum = 0;
            double sumSq = 0;
            for (float v : arr) {
                sum += v;
                sumSq += (double) v * v;
            }
            double mean = sum / n;
            double std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));

            Map<String, Double> stats = new HashMap<>();
            stats.put("p1", p1);
            stats.put("p99", p99);
            stats.put("min", min);
            stats.put("max", max);
            stats.put("mean", mean);
            stats.put("std", std);
            channelStats.add(stats);

            logger.debug(
                    "Channel {} stats: p1={}, p99={}, min={}, max={}, mean={}, std={}",
                    c,
                    String.format("%.4f", p1),
                    String.format("%.4f", p99),
                    String.format("%.4f", min),
                    String.format("%.4f", max),
                    String.format("%.4f", mean),
                    String.format("%.4f", std));
        }

        return channelStats;
    }
}
