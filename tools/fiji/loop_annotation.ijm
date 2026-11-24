// Capillary loop marking helpers for Fiji.
//
// Install: Plugins > Macros > Install... and choose this file
// (it needs installing again after Fiji restarts).
//
//   F5  save the loops marked on the open image  -> <image>_loops.csv next to it
//   F6  load previously saved loops back onto the open image, to continue or check
//
// Mark loops with the Multi-point tool (right-click the point tool to choose it).
// Click to add a point, Alt+click a point to remove it, drag to move it.

macro "Save loops [F5]" {
    if (nImages == 0) exit("Open an image first.");
    if (selectionType() != 10) exit("No points marked. Choose the Multi-point tool and click each loop first.");
    getSelectionCoordinates(xs, ys);
    path = getDirectory("image") + baseName() + "_loops.csv";
    if (File.exists(path)) {
        if (!getBoolean(baseName() + "_loops.csv already exists. Overwrite it?")) exit;
    }
    f = File.open(path);
    print(f, "x,y");
    for (i = 0; i < xs.length; i++)
        print(f, d2s(xs[i], 1) + "," + d2s(ys[i], 1));
    File.close(f);
    showMessage("Saved", xs.length + " loops saved to " + baseName() + "_loops.csv");
}

macro "Load loops [F6]" {
    if (nImages == 0) exit("Open an image first.");
    path = getDirectory("image") + baseName() + "_loops.csv";
    if (!File.exists(path)) exit("No saved loops for " + getTitle() + " yet.");
    lines = split(File.openAsString(path), "\n");
    n = lines.length - 1;
    xs = newArray(n);
    ys = newArray(n);
    for (i = 0; i < n; i++) {
        v = split(lines[i + 1], ",");
        xs[i] = parseFloat(v[0]);
        ys[i] = parseFloat(v[1]);
    }
    makeSelection("point", xs, ys);
    setTool("multipoint");
    showStatus("Loaded " + n + " loops");
}

function baseName() {
    title = getTitle();
    dot = lastIndexOf(title, ".");
    if (dot > 0) return substring(title, 0, dot);
    return title;
}
