// Copyright (c) Meta Platforms, Inc. and affiliates.
"use strict";
const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const examples = window.UNSTEP_EXAMPLES;
const videos = [$("original-video"), $("unstep-video")];
let model = "sf";
let currentId = examples[0].id;
let filtered = examples;
let playRequest = 0;
let pairWanted = false;

function setPlayLabel() {
  const playing = videos.some((video) => !video.paused);
  $("play-pair").querySelector("span").textContent = playing
    ? "Pause comparison"
    : "Play comparison";
  $("play-pair").querySelector("img").src =
    `media/icons/${playing ? "pause" : "play"}.svg`;
}

function pausePair() {
  playRequest += 1;
  pairWanted = false;
  videos.forEach((video) => video.pause());
  setPlayLabel();
}

async function playPair(restart = false) {
  const request = ++playRequest;
  pairWanted = true;
  if (restart)
    videos.forEach((video) => {
      video.currentTime = 0;
    });
  const results = await Promise.allSettled(videos.map((video) => video.play()));
  if (request !== playRequest) return;
  if (results.some((result) => result.status === "rejected")) pausePair();
  setPlayLabel();
}

function showExample(id) {
  pausePair();
  const example = filtered.find((item) => item.id === id) || filtered[0];
  currentId = example.id;
  $("example").value = example.id;
  $("sample-index").textContent =
    `${String(filtered.indexOf(example) + 1).padStart(2, "0")} / ${String(filtered.length).padStart(2, "0")}`;
  $("sample-prompt").textContent = example.prompt;
  $("reference-image").src = `media/i2v-${example.id}-reference.jpg`;
  $("reference-image").alt = `Reference image: ${example.prompt}`;
  $("reference-link").href = $("reference-image").getAttribute("src");
  $("reference-link").setAttribute(
    "aria-label",
    `Open reference image: ${example.prompt}`,
  );
  const name = model === "sf" ? "Self Forcing" : "Causal Forcing";
  $("original-label").textContent = name;
  $("unstep-label").textContent = `${name} + UnStep`;
  for (const [index, video] of videos.entries()) {
    const variant = `${model}_${index ? "unstep" : "original"}`;
    video.poster = `media/i2v-${example.id}-${variant}.jpg`;
    video.src = `media/i2v-${example.id}-${variant}.mp4`;
    video.setAttribute(
      "aria-label",
      `${index ? name + " + UnStep" : name}: ${example.prompt}`,
    );
    video.closest("figure").querySelector(".media-error").hidden = true;
    video.load();
  }
  $("previous").disabled = filtered.length < 2;
  $("next").disabled = filtered.length < 2;
}

function populateExamples() {
  filtered = examples.filter(
    (item) =>
      $("category").value === "all" || item.category === $("category").value,
  );
  $("example").replaceChildren(
    ...filtered.map((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.title;
      return option;
    }),
  );
  showExample(currentId);
}

$("category").addEventListener("change", populateExamples);
$("example").addEventListener("change", (event) =>
  showExample(event.target.value),
);
for (const button of document.querySelectorAll("[data-model]")) {
  button.addEventListener("click", () => {
    model = button.dataset.model;
    for (const other of document.querySelectorAll("[data-model]"))
      other.setAttribute("aria-pressed", String(other === button));
    showExample(currentId);
  });
}
for (const [id, direction] of [
  ["previous", -1],
  ["next", 1],
]) {
  $(id).addEventListener("click", () => {
    const index = filtered.findIndex((item) => item.id === currentId);
    showExample(
      filtered[(index + direction + filtered.length) % filtered.length].id,
    );
  });
}
$("play-pair").addEventListener("click", () =>
  videos.some((video) => !video.paused) ? pausePair() : playPair(true),
);
$("restart-pair").addEventListener("click", () => playPair(true));
videos.forEach((video) => {
  video.addEventListener("play", setPlayLabel);
  video.addEventListener("pause", setPlayLabel);
  video.addEventListener("error", () => {
    const error = video.closest("figure").querySelector(".media-error");
    error.hidden = false;
    error.querySelector("a").href = video.getAttribute("src");
    pausePair();
  });
});
// Correct loading/loop drift only during paired playback; native controls remain usable.
videos[0].addEventListener("timeupdate", () => {
  if (
    pairWanted &&
    videos.every((video) => !video.paused && video.readyState >= 3) &&
    Math.abs(videos[0].currentTime - videos[1].currentTime) > 0.25
  ) {
    videos[1].currentTime = videos[0].currentTime;
  }
});

const hero = $("hero-video");
let heroWanted = !reducedMotion.matches;
let heroVisible = true;
function heroLabel() {
  const label = hero.paused
    ? "Play background video"
    : "Pause background video";
  $("hero-toggle").setAttribute("aria-label", label);
  $("hero-toggle").title = label;
  $("hero-toggle").querySelector("img").src =
    `media/icons/${hero.paused ? "play" : "pause"}.svg`;
}
async function playHero() {
  if (!hero.querySelector("source").hasAttribute("src")) {
    hero.querySelector("source").src = hero.querySelector("source").dataset.src;
    hero.load();
  }
  try {
    await hero.play();
  } catch {
    /* Keep the poster when autoplay is blocked. */
  }
  heroLabel();
}
$("hero-toggle").addEventListener("click", () => {
  heroWanted = hero.paused;
  if (heroWanted) playHero();
  else hero.pause();
});
hero.addEventListener("play", heroLabel);
hero.addEventListener("pause", heroLabel);
hero.addEventListener("error", heroLabel);
reducedMotion.addEventListener("change", (event) => {
  heroWanted = !event.matches;
  if (heroWanted && heroVisible && !document.hidden) playHero();
  else hero.pause();
});
new IntersectionObserver(
  (entries) => {
    heroVisible = entries[0].isIntersecting;
    if (heroVisible && heroWanted && !document.hidden) playHero();
    else hero.pause();
  },
  { threshold: 0.05 },
).observe(hero);
new IntersectionObserver(
  (entries) => {
    if (!entries[0].isIntersecting) pausePair();
  },
  { threshold: 0.05 },
).observe(document.querySelector(".comparison"));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    hero.pause();
    pausePair();
  } else if (heroWanted && heroVisible) playHero();
});
populateExamples();
