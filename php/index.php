<?php

$csvFile = "plots.csv";

/*
plots.csv format:

canvas_name_1
canvas_name_2
canvas_name_3

(one canvas name per row)
*/

$plots = [];

if (!file_exists($csvFile)) {

    echo "<!doctype html>";
    echo "<html><head>";
    echo "<meta charset='utf-8'>";
    echo "<title>Subfolders</title>";

    echo "<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' rel='stylesheet'>";

    echo "</head><body class='p-4'>";
    $requestUri = $_SERVER['REQUEST_URI'] ?? '/';
    $parts = array_filter(explode('/', trim($requestUri, '/')));
    $accum = '';
    echo '<nav style="font-size:1rem; padding:8px 16px; background:#fff; border-bottom:1px solid #ddd;">';
    echo '<a href="/" class="text-decoration-none">
            🏠
        </a>';
    foreach (array_values($parts) as $i => $part) {
        $accum .= '/' . $part;
        echo ' / ';
        if ($i < count($parts) - 1) {
            echo '<a href="' . htmlspecialchars($accum) . '">' . htmlspecialchars($part) . '</a>';
        } else {
            echo '<span>' . htmlspecialchars($part) . '</span>';
        }
    }
    echo '</nav>';


    echo "<h5 class='mt-4'>Available subfolders</h5>";

    echo "<div class='list-group'>";

    foreach (glob("./*", GLOB_ONLYDIR) as $dir) {

        $name = basename($dir);

        echo "<a class='list-group-item list-group-item-action' href='" .
             htmlspecialchars($dir) .
             "'>" .
             htmlspecialchars($name) .
             "</a>";
    }

    echo "</div>";

    echo "</body></html>";

    exit;
}

if (($handle = fopen($csvFile, "r")) !== false) {

    while (($row = fgetcsv($handle)) !== false) {

        if (!empty($row[0])) {
            $plots[] = [
                "name" => trim($row[0])
            ];
        }
    }

    fclose($handle);
}

$rootfile = "../histos.root";

/* keep same defaults previously coming from JSON */
$cardW = 400;
$plotH = 300;


if (!file_exists($rootfile)) {
    // Mostra il breadcrumb prima di uscire
    ?>
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5/dist/css/bootstrap.min.css">
    </head><body>
    <div class="container-fluid px-3 py-2 bg-white border-bottom">
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11/font/bootstrap-icons.min.css">
      <?php
      $parts = array_values(array_filter(explode('/', trim($_SERVER['REQUEST_URI'] ?? '/', '/'))));
      $accum = '';
      foreach ($parts as $i => $part) {
          $accum .= '/' . $part;
          echo ' / ';
          if ($i < count($parts) - 1) {
              echo '<a href="' . htmlspecialchars($accum) . '">' . htmlspecialchars($part) . '</a>';
          } else {
              echo htmlspecialchars($part);
          }
      }
      ?>
    </div>
    <div class="container mt-4">
      <h5>Available subfolders</h5>
      <ul class="list-group">
      <?php foreach (glob("./*", GLOB_ONLYDIR) as $dir): ?>
        <li class="list-group-item">
          <a href="<?= htmlspecialchars(basename($dir)) ?>"><?= htmlspecialchars(basename($dir)) ?></a>
        </li>
      <?php endforeach; ?>
      </ul>
    </div>
    </body></html>
    <?php
    exit;
}


?>

<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>JSROOT Canvas Viewer</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

<style>
body {
    background: #f5f5f5;
}

.root-card {
    margin: 10px;
    width: var(--card-w, <?= (int)$cardW ?>px);
}

.root-canvas {
    width: 100%;
    height: var(--plot-h, <?= (int)$plotH ?>px);
    background: white;
    position: relative;
}
</style>
</head>

<body class="p-3">
<!-- Breadcrumb navigation -->
<nav class="navbar navbar-light bg-light border-bottom mb-3 px-3 py-2">

    <div class="d-flex align-items-center flex-wrap">

        <a href="/" class="text-decoration-none">
            🏠
        </a>

        <?php
        $uri   = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
        $parts = array_filter(explode('/', trim($uri, '/')));

        $accum = '';

        foreach ($parts as $i => $part) {

            $accum .= '/' . $part;

            echo '<span class="mx-2">/</span>';

            if ($i < count($parts) - 1) {

                echo '<a class="text-decoration-none" href="' .
                     htmlspecialchars($accum) .
                     '">' .
                     htmlspecialchars($part) .
                     '</a>';

            } else {

                echo '<span class="fw-light">' .
                     htmlspecialchars($part) .
                     '</span>';
            }
        }
        ?>

    </div>

</nav>

<h4>ROOT Canvas Viewer</h4>

<!-- CONTROLS -->
<div class="d-flex gap-4 mb-3 flex-wrap">

<div class="d-flex gap-2 align-items-center">
<label class="mb-0 small">Width</label>
<input type="range" id="w" min="200" max="900" value="<?= (int)$cardW ?>">
<span id="wVal" class="small text-muted"></span>
</div>

<div class="d-flex gap-2 align-items-center">
<label class="mb-0 small">Height</label>
<input type="range" id="h" min="150" max="800" value="<?= (int)$plotH ?>">
<span id="hVal" class="small text-muted"></span>
</div>

</div>

<!-- GRID -->
<div id="grid" class="d-flex flex-wrap"></div>

<script>
window.CONFIG = <?= json_encode($plots) ?>;
window.FILE   = <?= json_encode($rootfile) ?>;
</script>

<script type="module">

import {
    openFile,
    draw,
    resize
} from "https://root.cern/js/latest/modules/main.mjs";


import * as JSROOT from "https://root.cern/js/latest/modules/main.mjs";


JSROOT.gStyle.OptStat = 0;

/* ---------------- SIZE CONTROL ---------------- */

function applySize(w, h) {
    document.documentElement.style.setProperty('--card-w', w + 'px');
    document.documentElement.style.setProperty('--plot-h', h + 'px');

    document.querySelectorAll('.root-canvas').forEach(div => {
        try { resize(div); } catch(e) {}
    });
}

/* ---------------- MAIN ---------------- */

async function main() {

    const file = await openFile(window.FILE);
    const grid = document.getElementById("grid");

    for (const p of window.CONFIG) {

        const id = "c_" + p.name.replace(/\W/g, "_");

        const card = document.createElement("div");
        card.className = "card root-card";

        const fileName = p.name;

        console.log("PLOT ENTRY:", p);

        console.log("fileName:", fileName);
        console.log("id:", id);

        card.innerHTML = `
            <div class="card-header text-center fw-bold text-danger">
                ${fileName}
            </div>

            <div class="card-body p-1">

                <div id="${id}" class="root-canvas"></div>

                <!-- ACTION BAR -->
                <div class="d-flex justify-content-center gap-2 mt-2 flex-wrap">

                    <!-- JSROOT full view -->
                    <a class="btn btn-sm btn-primary"
                       target="_blank"
                       href="view.php?file=${encodeURIComponent(window.FILE)}&obj=${encodeURIComponent(fileName)}">
                        JSROOT ↗
                    </a>

                </div>

            </div>
        `;

        grid.appendChild(card);

        try {

            const obj = await file.readObject(fileName);

            JSROOT.gStyle.fOptStat = 0;

            let opt = "";
            console.log(opt);

            const painter = await draw(id, obj, opt);

            window._painters = window._painters || {};
            window._painters[id] = painter;

        } catch (e) {

            document.getElementById(id).innerHTML =
                `<div class="text-danger p-3">${e}</div>`;
        }
    }
}

/* ---------------- SLIDERS ---------------- */

const w = document.getElementById("w");
const h = document.getElementById("h");

function update() {

    const ww = w.value;
    const hh = h.value;

    document.getElementById("wVal").textContent = ww + "px";
    document.getElementById("hVal").textContent = hh + "px";

    applySize(ww, hh);
}

w.addEventListener("input", update);
h.addEventListener("input", update);

update();
main();

</script>

</body>
</html>
