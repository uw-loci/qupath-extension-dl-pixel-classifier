package qupath.ext.dlclassifier.ui;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;

/**
 * Reusable panel for selecting and configuring image channels.
 * <p>
 * This component provides:
 * <ul>
 *   <li>Multi-select channel list with color indicators</li>
 *   <li>Search/filter bar and bulk select for large channel counts</li>
 *   <li>Channel reordering with up/down buttons</li>
 *   <li>Normalization strategy selection per channel</li>
 *   <li>Per-channel normalization toggle for fluorescence images</li>
 *   <li>Real-time validation and feedback</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ChannelSelectionPanel extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(ChannelSelectionPanel.class);

    /** Threshold above which the search/filter toolbar is shown */
    private static final int SEARCH_BAR_THRESHOLD = 8;

    private final ListView<ChannelItem> availableList;
    private final ListView<ChannelItem> selectedList;
    private final ComboBox<ChannelConfiguration.NormalizationStrategy> normalizationCombo;
    private final CheckBox perChannelNormCheck;
    private final Label statusLabel;
    private final BooleanProperty validProperty = new SimpleBooleanProperty(false);

    // Search/filter toolbar components
    private final HBox searchToolbar;
    private final TextField searchField;

    // Master list backing the available channels (FilteredList wraps this)
    private final ObservableList<ChannelItem> masterAvailableList = FXCollections.observableArrayList();
    private final FilteredList<ChannelItem> filteredAvailableList = new FilteredList<>(masterAvailableList);

    private ImageServer<BufferedImage> currentServer;
    private int requiredChannelCount = -1; // -1 means any count is valid

    /**
     * Creates a new channel selection panel.
     */
    public ChannelSelectionPanel() {
        setSpacing(10);
        setPadding(new Insets(10));

        // Available channels list -- backed by filtered view of master list
        availableList = new ListView<>(filteredAvailableList);
        availableList.setCellFactory(lv -> new ChannelListCell());
        availableList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        availableList.setPrefHeight(150);
        TooltipHelper.install(
                availableList,
                "Image channels available for selection.\n" + "Multi-select with Ctrl+click or Shift+click.");

        // Selected channels list
        selectedList = new ListView<>();
        selectedList.setCellFactory(lv -> new ChannelListCell());
        selectedList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        selectedList.setPrefHeight(150);
        TooltipHelper.install(
                selectedList,
                "Channels that will be used as model input.\n"
                        + "Order matters: must match the order used during training.\n"
                        + "For RGB brightfield images, use Red/Green/Blue in order.\n"
                        + "For fluorescence, select only the channels relevant to your task.");

        // ---- Search/filter toolbar (hidden for small channel counts) ----
        searchField = new TextField();
        searchField.setPromptText("Filter channels...");
        searchField.setPrefWidth(180);
        HBox.setHgrow(searchField, Priority.ALWAYS);
        TooltipHelper.install(
                searchField, "Type to filter channels by name or index.\n" + "Case-insensitive substring match.");

        searchField.textProperty().addListener((obs, old, text) -> {
            filteredAvailableList.setPredicate(item -> {
                if (text == null || text.isEmpty()) return true;
                String lower = text.toLowerCase();
                return item.name().toLowerCase().contains(lower)
                        || String.valueOf(item.index()).contains(lower);
            });
        });

        Button selectAllVisibleBtn = new Button("Select All");
        selectAllVisibleBtn.setMinWidth(75);
        TooltipHelper.install(selectAllVisibleBtn, "Add all visible (filtered) channels to selected list");
        selectAllVisibleBtn.setOnAction(e -> addAllVisibleChannels());

        Button selectNoneBtn = new Button("Select None");
        selectNoneBtn.setMinWidth(85);
        TooltipHelper.install(selectNoneBtn, "Remove all channels from selected list");
        selectNoneBtn.setOnAction(e -> removeAllChannels());

        Button matchPatternBtn = new Button("Match Pattern...");
        matchPatternBtn.setMinWidth(110);
        TooltipHelper.install(
                matchPatternBtn,
                "Open a dialog to select channels by regex pattern.\n" + "Examples: CD\\d+, DAPI|CK.*");
        matchPatternBtn.setOnAction(e -> showPatternSelectDialog());

        searchToolbar = new HBox(8, searchField, selectAllVisibleBtn, selectNoneBtn, matchPatternBtn);
        searchToolbar.setAlignment(Pos.CENTER_LEFT);
        searchToolbar.setVisible(false);
        searchToolbar.setManaged(false);

        // Transfer buttons - fixed width to prevent text truncation
        double buttonWidth = 40;

        Button addButton = new Button(">");
        addButton.setMinWidth(buttonWidth);
        TooltipHelper.install(addButton, "Add selected channels to model input");
        addButton.setOnAction(e -> addSelectedChannels());

        Button removeButton = new Button("<");
        removeButton.setMinWidth(buttonWidth);
        TooltipHelper.install(removeButton, "Remove selected channels from model input");
        removeButton.setOnAction(e -> removeSelectedChannels());

        Button addAllButton = new Button(">>");
        addAllButton.setMinWidth(buttonWidth);
        TooltipHelper.install(addAllButton, "Add all available channels to model input");
        addAllButton.setOnAction(e -> addAllChannels());

        Button removeAllButton = new Button("<<");
        removeAllButton.setMinWidth(buttonWidth);
        TooltipHelper.install(removeAllButton, "Remove all channels from model input");
        removeAllButton.setOnAction(e -> removeAllChannels());

        VBox transferButtons = new VBox(5, addButton, removeButton, new Separator(), addAllButton, removeAllButton);
        transferButtons.setAlignment(Pos.CENTER);

        // Reorder buttons
        Button upButton = new Button("Up");
        TooltipHelper.install(upButton, "Move selected channel up in input order");
        upButton.setOnAction(e -> moveSelectedUp());

        Button downButton = new Button("Down");
        TooltipHelper.install(downButton, "Move selected channel down in input order");
        downButton.setOnAction(e -> moveSelectedDown());

        VBox reorderButtons = new VBox(5, upButton, downButton);
        reorderButtons.setAlignment(Pos.CENTER);

        // Layout for channel lists
        VBox availableBox = new VBox(5, new Label("Available Channels:"), availableList);
        VBox selectedBox = new VBox(5, new Label("Selected Channels:"), selectedList, reorderButtons);
        HBox.setHgrow(availableBox, Priority.ALWAYS);
        HBox.setHgrow(selectedBox, Priority.ALWAYS);

        HBox listsBox = new HBox(10, availableBox, transferButtons, selectedBox);
        listsBox.setAlignment(Pos.CENTER);

        // Normalization settings
        normalizationCombo =
                new ComboBox<>(FXCollections.observableArrayList(ChannelConfiguration.NormalizationStrategy.values()));
        normalizationCombo.setValue(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99);
        TooltipHelper.install(
                normalizationCombo,
                "Per-channel intensity normalization before model input:\n\n"
                        + "PERCENTILE_99 (recommended): Scale using 1st/99th percentiles.\n"
                        + "  Robust to outliers and hot pixels. Best for most images.\n\n"
                        + "MIN_MAX: Scale to [0,1] using channel min/max values.\n"
                        + "  Sensitive to outliers -- a single bright pixel affects the range.\n\n"
                        + "Z_SCORE: Subtract mean, divide by std deviation.\n"
                        + "  Common in deep learning; centers data around zero.\n\n"
                        + "FIXED_RANGE: Use a fixed intensity range (e.g. 0-255 for 8-bit).\n"
                        + "  Useful when consistent intensity scaling is required across images.");

        // Per-channel normalization toggle (hidden for small channel counts)
        perChannelNormCheck = new CheckBox("Per-channel normalization (normalize each channel independently)");
        perChannelNormCheck.setSelected(true);
        TooltipHelper.install(
                perChannelNormCheck,
                "When enabled, each channel is normalized independently\n"
                        + "using its own statistics. Recommended for fluorescence images\n"
                        + "where channels have different intensity ranges.\n\n"
                        + "When disabled, the same normalization range is applied\n"
                        + "across all channels (typical for brightfield RGB).");
        perChannelNormCheck.setVisible(false);
        perChannelNormCheck.setManaged(false);

        HBox normBox = new HBox(10, new Label("Normalization:"), normalizationCombo);
        normBox.setAlignment(Pos.CENTER_LEFT);

        // Status label
        statusLabel = new Label();
        statusLabel.setStyle("-fx-text-fill: #666;");

        // Add all components
        getChildren().addAll(searchToolbar, listsBox, normBox, perChannelNormCheck, statusLabel);

        // Update validity when selection changes
        selectedList.getItems().addListener((javafx.collections.ListChangeListener<ChannelItem>) c -> updateValidity());
    }

    /**
     * Populates the panel from an image data object.
     *
     * @param imageData the image data to get channels from
     */
    public void setImageData(ImageData<BufferedImage> imageData) {
        masterAvailableList.clear();
        selectedList.getItems().clear();
        searchField.clear();

        if (imageData == null || imageData.getServer() == null) {
            currentServer = null;
            updateSearchToolbarVisibility(0);
            updateValidity();
            return;
        }

        currentServer = imageData.getServer();
        List<ImageChannel> channels = currentServer.getMetadata().getChannels();

        for (int i = 0; i < channels.size(); i++) {
            ImageChannel channel = channels.get(i);
            Color color = javafxColorFromAwtColor(channel.getColor());
            masterAvailableList.add(new ChannelItem(i, channel.getName(), color));
        }

        updateSearchToolbarVisibility(channels.size());
        logger.debug("Loaded {} channels from image", channels.size());
        updateValidity();
    }

    /**
     * Sets the required number of channels for validation.
     *
     * @param count required channel count, or -1 for any
     */
    public void setRequiredChannelCount(int count) {
        this.requiredChannelCount = count;
        updateValidity();
    }

    /**
     * Gets the current channel configuration.
     *
     * @return channel configuration based on current selections
     */
    public ChannelConfiguration getChannelConfiguration() {
        List<Integer> indices =
                selectedList.getItems().stream().map(ChannelItem::index).collect(Collectors.toList());

        List<String> names =
                selectedList.getItems().stream().map(ChannelItem::name).collect(Collectors.toList());

        int bitDepth = currentServer != null ? currentServer.getPixelType().getBitsPerPixel() : 8;

        return ChannelConfiguration.builder()
                .selectedChannels(indices)
                .channelNames(names)
                .bitDepth(bitDepth)
                .normalizationStrategy(normalizationCombo.getValue())
                .perChannelNormalization(perChannelNormCheck.isSelected())
                .build();
    }

    /**
     * Sets the channel configuration from a classifier's expected channels.
     *
     * @param channelNames the expected channel names
     */
    public void setExpectedChannels(List<String> channelNames) {
        if (channelNames == null || channelNames.isEmpty()) {
            return;
        }

        // Try to match channels by name
        selectedList.getItems().clear();
        for (String name : channelNames) {
            for (ChannelItem item : masterAvailableList) {
                if (item.name().equalsIgnoreCase(name)) {
                    selectedList.getItems().add(item);
                    break;
                }
            }
        }

        setRequiredChannelCount(channelNames.size());
        updateValidity();
    }

    /**
     * Auto-selects all available channels. Useful for brightfield images
     * where channel selection is not meaningful.
     */
    public void selectAllChannels() {
        addAllChannels();
    }

    /**
     * Gets the number of available channels.
     *
     * @return number of channels in the available list (total, not filtered)
     */
    public int getAvailableChannelCount() {
        return masterAvailableList.size();
    }

    /**
     * Gets whether the current selection is valid.
     *
     * @return true if valid
     */
    public boolean isValid() {
        return validProperty.get();
    }

    /**
     * Gets the valid property for binding.
     *
     * @return the valid property
     */
    public BooleanProperty validProperty() {
        return validProperty;
    }

    /**
     * Gets the selected channel count.
     *
     * @return number of selected channels
     */
    public int getSelectedChannelCount() {
        return selectedList.getItems().size();
    }

    /**
     * Auto-configures the panel based on detected image type and channel count.
     * <ul>
     *   <li>channelCount &lt;= 8: hides search bar, auto-selects all channels</li>
     *   <li>Brightfield: hides per-channel checkbox, unchecks per-channel norm</li>
     *   <li>Fluorescence/other with &gt;8 channels: shows search bar, per-channel checked</li>
     * </ul>
     *
     * @param imageType the image type from QuPath
     * @param channelCount total number of channels in the image
     */
    public void autoConfigureForImageType(ImageData.ImageType imageType, int channelCount) {
        boolean isBrightfield = imageType == ImageData.ImageType.BRIGHTFIELD_H_E
                || imageType == ImageData.ImageType.BRIGHTFIELD_H_DAB
                || imageType == ImageData.ImageType.BRIGHTFIELD_OTHER;

        // Search toolbar visibility based on channel count
        updateSearchToolbarVisibility(channelCount);

        // Per-channel normalization: show for >3 channels, hide for brightfield
        if (isBrightfield || channelCount <= 3) {
            perChannelNormCheck.setSelected(false);
            perChannelNormCheck.setVisible(false);
            perChannelNormCheck.setManaged(false);
        } else {
            perChannelNormCheck.setSelected(true);
            perChannelNormCheck.setVisible(true);
            perChannelNormCheck.setManaged(true);
        }

        // Auto-select all channels for small channel counts
        if (channelCount <= SEARCH_BAR_THRESHOLD) {
            addAllChannels();
        }

        logger.debug(
                "Auto-configured for imageType={}, channelCount={}: searchBar={}, perChannel={}",
                imageType,
                channelCount,
                channelCount > SEARCH_BAR_THRESHOLD,
                perChannelNormCheck.isSelected());
    }

    /**
     * Sets a specific channel at a given position in the selected list by name.
     * Used by the InferenceDialog channel mapping to manually override a mapping.
     *
     * @param position zero-based position in the selected list
     * @param channelName name of the channel to place at that position
     */
    public void setSelectedChannelByName(int position, String channelName) {
        if (position < 0 || position >= selectedList.getItems().size()) {
            logger.warn("Invalid position {} for setSelectedChannelByName", position);
            return;
        }

        // Find the channel in the master list
        ChannelItem replacement = null;
        for (ChannelItem item : masterAvailableList) {
            if (item.name().equals(channelName)) {
                replacement = item;
                break;
            }
        }

        if (replacement == null) {
            logger.warn("Channel '{}' not found in available channels", channelName);
            return;
        }

        // Remove duplicates of this channel if already selected elsewhere
        selectedList.getItems().remove(replacement);

        // Ensure position is still valid after potential removal
        if (position >= selectedList.getItems().size()) {
            selectedList.getItems().add(replacement);
        } else {
            selectedList.getItems().set(position, replacement);
        }
    }

    /**
     * Sets the normalization strategy combo box value.
     *
     * @param strategy the normalization strategy to select
     */
    public void setNormalizationStrategy(ChannelConfiguration.NormalizationStrategy strategy) {
        if (strategy != null) {
            normalizationCombo.setValue(strategy);
        }
    }

    // ---- Private helper methods ----

    private void updateSearchToolbarVisibility(int channelCount) {
        boolean show = channelCount > SEARCH_BAR_THRESHOLD;
        searchToolbar.setVisible(show);
        searchToolbar.setManaged(show);
    }

    private void addSelectedChannels() {
        List<ChannelItem> toAdd =
                new ArrayList<>(availableList.getSelectionModel().getSelectedItems());
        for (ChannelItem item : toAdd) {
            if (!selectedList.getItems().contains(item)) {
                selectedList.getItems().add(item);
            }
        }
    }

    private void removeSelectedChannels() {
        List<ChannelItem> toRemove =
                new ArrayList<>(selectedList.getSelectionModel().getSelectedItems());
        selectedList.getItems().removeAll(toRemove);
    }

    private void addAllChannels() {
        for (ChannelItem item : masterAvailableList) {
            if (!selectedList.getItems().contains(item)) {
                selectedList.getItems().add(item);
            }
        }
    }

    /**
     * Adds all currently visible (filtered) channels to the selected list.
     */
    private void addAllVisibleChannels() {
        for (ChannelItem item : filteredAvailableList) {
            if (!selectedList.getItems().contains(item)) {
                selectedList.getItems().add(item);
            }
        }
    }

    private void removeAllChannels() {
        selectedList.getItems().clear();
    }

    private void moveSelectedUp() {
        int index = selectedList.getSelectionModel().getSelectedIndex();
        if (index > 0) {
            ChannelItem item = selectedList.getItems().remove(index);
            selectedList.getItems().add(index - 1, item);
            selectedList.getSelectionModel().select(index - 1);
        }
    }

    private void moveSelectedDown() {
        int index = selectedList.getSelectionModel().getSelectedIndex();
        if (index >= 0 && index < selectedList.getItems().size() - 1) {
            ChannelItem item = selectedList.getItems().remove(index);
            selectedList.getItems().add(index + 1, item);
            selectedList.getSelectionModel().select(index + 1);
        }
    }

    private void updateValidity() {
        int selected = selectedList.getItems().size();
        int total = masterAvailableList.size();
        boolean valid;

        if (selected == 0) {
            statusLabel.setText("Please select at least one channel");
            statusLabel.setStyle("-fx-text-fill: #cc0000;");
            valid = false;
        } else if (requiredChannelCount > 0 && selected != requiredChannelCount) {
            statusLabel.setText(
                    String.format("Selected %d channels (classifier expects %d)", selected, requiredChannelCount));
            statusLabel.setStyle("-fx-text-fill: #cc0000;");
            valid = false;
        } else {
            // Show "X of Y" when there are more available than selected
            if (total > selected) {
                statusLabel.setText(String.format("%d of %d channels selected", selected, total));
            } else {
                statusLabel.setText(String.format("%d channel(s) selected", selected));
            }
            statusLabel.setStyle("-fx-text-fill: #006600;");
            valid = true;
        }

        validProperty.set(valid);
    }

    private void showPatternSelectDialog() {
        TextInputDialog dialog = new TextInputDialog();
        dialog.setTitle("Select by Pattern");
        dialog.setHeaderText("Enter a regex pattern to match channel names");
        dialog.setContentText("Pattern:");
        if (getScene() != null && getScene().getWindow() != null) {
            dialog.initOwner(getScene().getWindow());
        }
        DialogOwner.own(dialog);
        dialog.showAndWait().ifPresent(patternStr -> {
            if (patternStr.isEmpty()) return;
            try {
                Pattern regex = Pattern.compile(patternStr, Pattern.CASE_INSENSITIVE);
                int matched = 0;
                for (ChannelItem item : masterAvailableList) {
                    if (regex.matcher(item.name()).find()
                            && !selectedList.getItems().contains(item)) {
                        selectedList.getItems().add(item);
                        matched++;
                    }
                }
                statusLabel.setText(String.format("Pattern matched %d new channel(s)", matched));
                statusLabel.setStyle("-fx-text-fill: #006600;");
            } catch (PatternSyntaxException e) {
                statusLabel.setText("Invalid regex: " + e.getMessage());
                statusLabel.setStyle("-fx-text-fill: #cc0000;");
            }
        });
    }

    private Color javafxColorFromAwtColor(Integer argb) {
        if (argb == null) {
            return Color.GRAY;
        }
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8) & 0xFF;
        int b = argb & 0xFF;
        return Color.rgb(r, g, b);
    }

    /**
     * Represents a channel item in the list.
     */
    public record ChannelItem(int index, String name, Color color) {
        @Override
        public String toString() {
            return String.format("[%d] %s", index, name);
        }
    }

    /**
     * Custom cell renderer for channel items.
     */
    private static class ChannelListCell extends ListCell<ChannelItem> {
        private final HBox content;
        private final Rectangle colorBox;
        private final Label nameLabel;

        public ChannelListCell() {
            colorBox = new Rectangle(16, 16);
            colorBox.setStroke(Color.BLACK);
            colorBox.setStrokeWidth(1);

            nameLabel = new Label();

            content = new HBox(8, colorBox, nameLabel);
            content.setAlignment(Pos.CENTER_LEFT);
        }

        @Override
        protected void updateItem(ChannelItem item, boolean empty) {
            super.updateItem(item, empty);

            if (empty || item == null) {
                setGraphic(null);
            } else {
                colorBox.setFill(item.color());
                nameLabel.setText(item.toString());
                setGraphic(content);
            }
        }
    }
}
