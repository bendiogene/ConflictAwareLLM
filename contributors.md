# Contributors



This repo was copied at some point from an internal directory, so the commit history *does not hence reflect contributions*. 



Contributions are therefore listed here.



## Zied Ben Houidi: Design, conceptualization, initial bootstrapping, supervision and later extended development

- Conceived the parallel between human cognitive dissonance and LLM's lack thereof

- Designed the human-inspired experimental approach (historical extraction, dissonance awareness and targeted update strategies differentiating dissonant and non-dissonant updates)

- Provided initial experimental framework and basic codebase (e.g. basic FT for facts/counter-facts, basic evaluation function, basic activation/gradient extraction)

- Supervised implementation and structured experimental workflow

- Later extended the work (e.g. new features for dissonance awareness (output probabilities), harmonized placement strategies, larger models and additional experimental analysis and results)

- Wrote the paper, created figures, and documentation



## Simone Clemente: Core end-to-end implementation and first iteration of experimentation

- Developed a robust, clear and maintainable end-to-end pipeline enabling systematic evaluation, including:

    - Rewriting and optimization of evaluation mechanisms

    - Full implementation of historical activation/gradient collection system

    - Development and evaluation of dissonance awareness (classifiers)

    - Development of custom fine-tuning with modular placement strategies

- Made practical implementation choices that simplified the approach (e.g., neuron-level rather than weight-level, simplified gradient and activation features)

- Conducted initial full round of experiments on base model