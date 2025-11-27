---
layout: default
title: Discovering State-of-the-art Reinforcement Learning Algorithms
description: We show that it is possible to automatically discover
 a state-of-the-art reinforcement learning (RL) algorithm that outperforms
 manually-designed ones across a variety of challenging benchmarks.
theme: jekyll-theme-cayman
---

<style>
    /* --- HEADER STYLES --- */
    .page-header {
        background-image: url('assets/cover.jpg');
        background-size: cover;
        background-position: center center;
        box-shadow: inset 0 0 0 1000px rgba(0,0,0,0.3);
    }
    .project-name {
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0,0,0,0.7);
        margin-bottom: 15px;
    }
    .project-tagline {
        background-color: rgba(0, 0, 0, 0.45);
        padding: 15px 25px;
        margin-top: 20px;
        border-radius: 8px;
        color: #ffffff;
        font-size: 1.4rem;
        line-height: 1.5;
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(5px);
    }

    /* --- GENERAL PAGE STYLES --- */
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 0 20px;
    }
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .text-center { text-align: center; }
    .authors { font-size: 1.1em; margin-bottom: 10px; }
    .affiliations { font-size: 1em; color: #555; margin-bottom: 20px; }
    .links { margin-top: 20px; margin-bottom: 30px; }
    .links a {
        margin: 0 10px;
        font-size: 1.1em;
        font-weight: bold;
        text-decoration: none;
        padding: 8px 12px;
        border-radius: 5px;
        background-color: #f6f8fa;
        border: 1px solid #d1d5da;
        color: #0366d6;
    }
    h2 {
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
        margin-top: 40px;
    }

    /* --- SIDE-BY-SIDE RESULTS COMPONENT --- */
    .static-wrapper {
        display: flex;
        flex-direction: row;
        height: 55vh;
        min-height: 600px;
        max-height: 1000px;
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        overflow: hidden;
        width: 95vw;
        max-width: 1600px;
        margin-left: auto;
        margin-right: auto;
        margin-top: 40px;
        margin-bottom: 40px;
        position: relative;
        z-index: 1;
        left: 50%;
        transform: translateX(-50%);
    }

    .text-col {
        flex: 1;
        min-width: 300px;
        max-width: 450px;
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
        overflow-y: auto;
        padding: 30px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 20px;
    }

    .stage-description {
        padding: 20px 25px;
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #ccc;
        cursor: pointer;
        transition: all 0.2s ease-out;
        opacity: 0.7;
        pointer-events: auto;
    }

    .stage-description:hover {
        opacity: 0.9;
        transform: translateX(5px);
    }

    .stage-description.active-step {
        opacity: 1;
        border-left-color: #1a9f5d;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transform: scale(1.02);
    }

    .stage-description h3 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.25rem;
        color: #24292e;
    }

    .stage-description p {
        margin: 0;
        font-size: 1rem;
        color: #586069;
        line-height: 1.5;
    }

    .figure-col {
        flex: 3;
        position: relative;
        background-color: #fff;
        overflow: hidden;
    }

    .figure-state {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        opacity: 0;
        transition: opacity 0.4s ease-in-out;
        pointer-events: none;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .figure-state.active {
        opacity: 1;
        pointer-events: auto;
        cursor: zoom-in;
    }

    .figure-state.active:hover img {
        transform: scale(1.01);
    }

    .figure-state img {
        max-width: 92%;
        max-height: 92%;
        width: auto; height: auto;
        object-fit: contain;
        display: block;
        transition: transform 0.2s ease;
    }

    /* --- IMAGE MODAL STYLES --- */
    #image-modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.9);
        backdrop-filter: blur(5px);
        flex-direction: column;
        justify-content: center;
        align-items: center;
        animation: fadeIn 0.3s;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    #image-modal img {
        max-width: 90%;
        max-height: 85vh;
        object-fit: contain;
        border-radius: 4px;
        box-shadow: 0 5px 30px rgba(0,0,0,0.5);
    }

    #modal-close {
        position: absolute;
        top: 20px;
        right: 30px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        transition: 0.3s;
        cursor: pointer;
    }
    #modal-close:hover { color: #bbb; }

    @media (max-width: 1000px) {
        .static-wrapper {
            flex-direction: column-reverse;
            height: auto;
            width: 100%;
            left: auto;
            transform: none;
        }
        .text-col {
            width: 100%; max-width: none;
            height: auto;
            padding: 20px;
            flex-direction: row;
            overflow-x: auto;
            justify-content: flex-start;
            align-items: stretch;
        }
        .stage-description {
            min-width: 260px;
            margin-right: 15px;
        }
        .figure-col {
            height: 50vh; min-height: 300px;
        }
    }
</style>
<div class="container">

<div class="text-center">
    <p class="authors">
        Junhyuk Oh<sup>*</sup>,
        Iurii Kemaev<sup>*†</sup>,
        Greg Farquhar<sup>*</sup>,
        Dan A. Calian<sup>*</sup>,
        Matteo Hessel,<br>
        Luisa Zintgraf,
        Satinder Singh,
        Hado van Hasselt,
        David Silver
    </p>

    <p class="affiliations">
        Google DeepMind<br>
        <sup>*</sup>Equal contribution. <sup>†</sup>Engineering lead.
    </p>

    <p>Published in <strong>Nature</strong> (2025)</p>

    <div class="links">
        <a href="https://www.nature.com/articles/s41586-025-09761-x">Paper</a>
        <a href="https://github.com/google-deepmind/disco_rl">Code</a>
        <a href="#citation">Citation</a>
    </div>
</div>

<h2>Intro</h2>

<p>The field of artificial intelligence (AI) has been revolutionized by replacing hand-crafted components with those learned from data and experience. The next natural step is to allow the learning algorithms themselves to be learned, from experience.</p>

<img src="assets/intro_timeline.png" style="width: 100%;">

<p>Many of the most successful AI agents are based on reinforcement learning (RL), in which agents learn by interacting with environments, achieving numerous landmarks including the mastery of complex competitive games such as Go, chess, and StarCraft.</p>

<p>Traditional RL algorithms can be written down with equations and implemented in code. They are designed by human experts in a laborious process of trial and error, guided by experiment, theory, and human intuitions.</p>

<p>On the other hand, our discovered rule, which we call <b>DiscoRL</b>, is represented by a neural network which can be much more flexible than simple mathematical equations. Instead of being hand-crafted, it is learned by an automated process using the experience of many agents interacting with diverse and complex environments.</p>

<p>DiscoRL outperforms many existing RL algorithms on a variety of benchmarks and becomes stronger with more environments used for discovery.</p>

<h2>Method</h2>

<img src="assets/method.png" style="width:25%;float:left;margin:30px">

<p>RL agents typically make predictions that are useful for learning. The semantics of them are determined by their update rules such as the value of a certain action. In our framework, the agent makes additional predictions without pre-defined semantics to open up the possibility to discover entirely new prediction semantics.</p>

<p>RL agents optimise their policies and predictions using a loss function. This function depends on the agent’s own predictions, as well as the rewards it receives while interacting with its environment.</p>

<p>Instead of manually defining the loss function with equations, we use a neural network, called ‘meta-network’, to define the loss function for the prediction and policy. The meta-network is randomly initialised, which in turn initially acts as a random update rule. We let the meta-learning process optimise the meta-network to gradually discover more efficient update rules.</p>

<img src="assets/method_2.png" style="width: 95%;">

<p>In order to discover a strong update rule from experience, we create a large population of agents, each of which interacts with its own environment. Each agent uses the shared meta-network to update their predictions and policies. We then estimate the performance of them to calculate a meta-gradient, which gives how to adjust the meta-network that would lead to a better performance. Over time, the discovered rule becomes a stronger and faster RL algorithm. After the discovery process is complete, the discovered rule can be used to train new agents in unseen environments.</p>

<p>In order to scale up the discovery process, we developed a meta-learning framework that can handle hundreds of parallel agents. To improve fault-tolerance, and to facilitate our research, we ensured that all aspects of agents, environments, and meta-learning were deterministic and checkpointable thus providing full reproducibility. We also implemented a number of optimisations to handle the compute intense meta-gradients, including mixed-mode differentiation, recursive gradient checkpointing, mixed precision training, and pre-emptive parameter offloading<sup><a href="#mixflowmg">(+)</a></sup>.</p>

<p><a id="mixflowmg">(<sup><b>+</b></sup>)</a> Kemaev, I., Calian, D. A., Zintgraf, L. M., Farquhar, G. & van Hasselt, H. <em>Scalable meta-learning via mixed-mode differentiation.</em> International conference on machine learning (2025)</p>

<h2>Results</h2>

<p>After large-scale meta-learning, we find that DiscoRL outperforms the state-of-the-art or performs competitively on a number of challenging benchmarks.</p>

<div class="static-wrapper" id="interactive-results">
    <div class="text-col">
        <div class="stage-description active-step" data-step="1">
            <h3>1. Meta-learn Disco57</h3>
            <p>We start by training the Disco57 update rule on a diverse set of standard Atari57 environments.</p>
        </div>
        <div class="stage-description" data-step="2">
            <h3>2. Evaluate Disco57</h3>
            <p>Disco57 is evaluated zero-shot on completely unseen ProcGen and DMLab-30 domains to measure generalization capabilities.</p>
        </div>
        <div class="stage-description" data-step="3">
            <h3>3. Expand training set: meta-learn Disco103</h3>
            <p>We iteratively expand the training domains with the more challenging ProcGen and DMLab-30 to meta-learn Disco103.</p>
        </div>
        <div class="stage-description" data-step="4">
            <h3>4. Evaluate on unseen domains</h3>
            <p>We evaluate both Disco57 and Disco103 on unseen domains: Crafter, Nethack, Sokoban.</p>
        </div>
    </div>
    <div class="figure-col" id="figure-container">
        <div class="figure-state active" data-state="1">
            <img class="img-align-4" src="assets/result_1.png">
        </div>
        <div class="figure-state" data-state="2">
            <img class="img-align-4" src="assets/result_2.png">
        </div>
        <div class="figure-state" data-state="3">
            <img class="img-align-4" src="assets/result_3.png">
        </div>
        <div class="figure-state" data-state="4">
            <img id="figure-4" src="assets/result_4_2.png">
        </div>
    </div>
</div>

<img src="assets/results_scaling.png" style="width:40%;float:right;margin:30px">
<p>DiscoRL <b>generalises</b>, performing well in environments which were not used for discovery, and which have radically different observations and action spaces. DiscoRL also generalises when used to train agents with much more parameters and data than those used for discovery.</p>

<p>The discovery process <b>scales</b>, increasing performance as we increase the number, diversity, and complexity of training environments, as well as the overall amount of experience consumed.</p>

<img src="assets/results_analysis.png" style="width: 95%;margin:30px">

<p>We find that the discovered predictions capture novel semantics, identifying important features about upcoming events on moderate time-scales, such as future policy entropies and large-reward events. See the <a href="https://www.nature.com/articles/s41586-025-09761-x">manuscript</a> for more details.</p>

The overall results suggest that the design of RL algorithms may, in the future, be led by automated methods that can scale effectively with data and compute.

<h2 id="citation">Citation</h2>

<pre class="highlight"><code>
@Article{DiscoRL2025,
  author  = {Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},
  journal = {Nature},
  title   = {Discovering State-of-the-art Reinforcement Learning Algorithms},
  year    = {2025},
  doi     = {10.1038/s41586-025-09761-x}
}
</code></pre>

<hr>

<h2>Code Availability</h2>

We provide the meta-training and evaluation code, with the meta-parameters of Disco103, under an open source Apache 2.0 licence, <a href="https://github.com/google-deepmind/disco_rl">on GitHub</a>.

<h2>Acknowledgements</h2>

<p>We thank Sebastian Flennerhag, Zita Marinho, Angelos Filos, Surya Bhupatiraju, Andras György, and Andrei A. Rusu for their feedback and discussions about related ideas. We also thank Blanca Huergo Muñoz, Manuel Kroiss, and Dan Horgan for their help with the engineering infrastructure. Finally, we thank Raia Hadsell, Koray Kavukcuoglu, Nando de Freitas, and Oriol Vinyals for their high-level feedback on the project, and Simon Osindero and Doina Precup for their feedback on the early version of this work.</p>

<h2>License and disclaimer</h2>

<p>Copyright 2025 Google LLC</p>

<p>All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
<a href="https://www.apache.org/licenses/LICENSE-2.0">https://www.apache.org/licenses/LICENSE-2.0</a></p>

<p>All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">https://creativecommons.org/licenses/by/4.0/legalcode</a></p>

<p>Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.</p>

<p>This is not an official Google product.</p>

<hr>

</div>

<div id="image-modal">
    <span id="modal-close">&times;</span>
    <img id="modal-img" src="">
</div>

<!----- Javascript ----->
<script>
    let currentStep = 1;
    let isSwitching = false;
    let scrollAccumulator = 0;
    const SCROLL_THRESHOLD = 200;

    document.addEventListener("DOMContentLoaded", () => {
        const wrapper = document.getElementById('interactive-results');
        const descriptions = document.querySelectorAll('.stage-description');
        const figureStates = document.querySelectorAll('.figure-state');
        const modal = document.getElementById('image-modal');
        const modalImg = document.getElementById('modal-img');
        const modalClose = document.getElementById('modal-close');

        if (!wrapper) return;

        // Open modal on figure click
        document.querySelectorAll('.figure-state img').forEach(img => {
            img.addEventListener('click', (e) => {
                // Only open if parent is active (double check, though CSS handles most of it)
                if (img.parentElement.classList.contains('active')) {
                    modal.style.display = "flex";
                    modalImg.src = img.src;
                }
            });
        });
        // Close modal
        modalClose.addEventListener('click', () => { modal.style.display = "none"; });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = "none";
        });

        const switchStage = (targetStep, shouldScroll = true) => {
            targetStep = Math.max(1, Math.min(4, targetStep));
            currentStep = targetStep;

            // Update Text
            descriptions.forEach(d => d.classList.remove('active-step'));
            const activeDesc = document.querySelector(`.stage-description[data-step="${targetStep}"]`);
            if (activeDesc) {
                 activeDesc.classList.add('active-step');
                 if (shouldScroll) {
                    activeDesc.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                 }
            }

            // Update Figures
            figureStates.forEach(s => s.classList.remove('active'));
            const activeFig = document.querySelector(`.figure-state[data-state="${targetStep}"]`);
            if (activeFig) activeFig.classList.add('active');

        };

        wrapper.addEventListener('wheel', (e) => {
            if (isSwitching) {
                e.preventDefault();
                e.stopPropagation();
                return;
            }

            const direction = Math.sign(e.deltaY);
            const nextStep = currentStep + direction;

            // Check if the next step is within our valid range (1 to 4)
            if (nextStep >= 1 && nextStep <= 4) {
                e.preventDefault();
                e.stopPropagation();

                // Logic for "sticky" scrolling sensitivity
                if (scrollAccumulator !== 0 && Math.sign(e.deltaY) !== Math.sign(scrollAccumulator)) {
                    scrollAccumulator = 0;
                }

                scrollAccumulator += e.deltaY;

                if (Math.abs(scrollAccumulator) >= SCROLL_THRESHOLD) {
                    const direction = Math.sign(scrollAccumulator);
                    switchStage(currentStep + direction);
                    isSwitching = true;
                    scrollAccumulator = 0;
                    setTimeout(() => { isSwitching = false; }, 900);
                }
            }
            // If nextStep is < 1 or > 4, and isSwitching is false,
            // we let the browser handle the event (page scrolls naturally).
        }, { passive: false });

        descriptions.forEach(desc => {
            desc.addEventListener('click', () => {
                switchStage(parseInt(desc.getAttribute('data-step')));
            });
        });

        switchStage(1, false);
    });

    // Close modal on ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === "Escape") document.getElementById('image-modal').style.display = "none";
    });

</script>
