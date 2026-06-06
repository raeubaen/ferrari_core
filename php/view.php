<?php
$obj  = $_GET["obj"] ?? null;

if (!$obj) {
    die("Invalid request");
}
?>

<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>JSROOT View</title>

<script type="module">
import { openFile, draw } from "https://root.cern/js/latest/modules/main.mjs";

async function main() {
    const file = await openFile("../canvases.root");
    const obj  = await file.readObject("<?= htmlspecialchars($obj) ?>");

    await draw("canvas", obj, "hist;autozoom");
}

main();
</script>

<style>
body { margin:0; background:#111; }
#canvas { width:100vw; height:100vh; }
</style>

</head>

<body>
<div id="canvas"></div>
</body>
</html>
