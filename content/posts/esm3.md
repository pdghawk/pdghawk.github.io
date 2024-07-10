+++
title = 'ESM3 is out: A first summary'
date = 2024-07-02T11:55:56-04:00
tags = ['protein structure', 'language models']
draft = false
+++

On June 25th 2024 the ESM authors placed a [preprint](https://evolutionaryscale-public.s3.us-east-2.amazonaws.com/research/esm3.pdf) online (also on [BioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1.full.pdf)[^esm3] on July 2nd) titled "Simulating 500 million years of evolution with a language model".  

## Key takeaways
1. Using a masked language model, not an autoregressive model, with multiple tracks (sequence, structure) allows one to generate samples from any track 
2. Diffusion is not used for sampling structures, unlike Alphafold3[^af3]. The tokenization, and generative procedure is performed via an attentive VQVAE model with iterative decoding scheme.
3. Three model sizes were trained: 1.4B, 7B, 98B, with increased performance with more parameters. 1.4B is available for non-commercial purposes ([license](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md)) and code is on [Github](https://github.com/evolutionaryscale/esm).
5. Proteins can be generated conditioned on a subset of the protein's sequence (e.g a few residues only), structure, or even secondary structure, solvent exposure, function.
6. An example with experimental verification shows a design of a novel GFP (Green Fluorescent Protein) protein, by fixing a small number of key residues and generating the remaining protein. Performed using a novel "chain of thought" iterative generative process.

## Multi-track masked language model
ESM3 is designed in many ways like a typical BERT[^bert] model, with the major exception that there are multiple input "tracks". One for sequence, one for structure, and one for function. The structure and sequence tracks as the two most important in terms of typical use case I might expect, so let's start there. One when is only building a language model for protein sequence, the task is to predict the token (amino acid) at masked positions in the sequence. This leads to such models understanding the "grammar" of proteins, and the models representations have been shown to be useful for a range of downstream tasks including predicting how the protein folds in 3D space, see e.g.[^esmfold].

<!---![image](/images/esm3_post/three_track_model.png)--->

For structure things are a little more complicated - how should one construct tokens from structure? How does one decode from tokens back to structure? Interestingly Alphafold3[^af3] also tokenized structures and generates structures. In the Alphafold3 case the structure generation is via diffusion which is different from what is done in ESM3. We'll get into more details on this later, for now just take it for granted that one can encode protein structure into series of tokens similarly to text.

The important thing is that when training this masked language model on multiple inputs one masks all tracks of incoming information (with some rules), and asks the model to predict the tokens at the missing sites. This allows one then to receive information from any subset of the different input tracks, learning a joint distribution over them. This is a key difference to prior modeling approaches that typically learn representations of only sequence, or try to predict structure.

The structural information is supplemented with secondary structure and surface exposure information, which is easy to calculate but probably gives the model a boost as well as allowing for better conditioning for protein engineering/design, more on that later. 

### So how is it generative?
Generating new sequences works in much the same style as typical BERT models: the model predicts what tokens should be at each position in the input, and typically at masked input positions for in-filling or generative modeling. If one takes a typical BERT model and attempts to generate new data one may not expect a great new sample unless one provides a lot of context. If one thinks about a typical BERT type model for human language, the job of the model is to predict the likely token at the masked position. Now, if everything is masked, the model is unlikely to give me anything useful as there is no context, but filling in a few words works well. In the case of ESM3 and its multi-track nature, one can provide only the structure and use it as context in one track to generate (or predict) the sequence (without any of the sequence given in the input), or vice versa. This allows the model to perform:
 1. Protein folding : predict structure given only the sequence, or 
 2. Protein folding with additional context : predict structure given *some* information about structure 
 3. Inverse folding for protein regions: Given some structural and sequence information fill in missing regions of a protein sequence. 

On point 3 it is worth noting that the ESM3 paper does not give results about true inverse folding - predicting the sequence from structure alone. They have previously released ESM-IF[^esmif], an inverse folding specific model, and they use that in the ESM3 paper in several places. They do train a small (200M parameter) IF model with ESM3 techniques however, which they use for data augmentation.

One nice thing about about the multi-track generative approach is that one can also give some information about the thing one is trying to generate. That is, rather than asking the model to generate the entire sequence just from structure, I could give the sequence for some region of the protein (leaving the remainder masked), and the structure of some related protein, and ask the model to generate a sequence for the whole thing based on that information. This can be useful for hinting sequence regions or vice versa for instance. As we will see later performing generative modeling in a chain of steps can be quite powerful: namely predicting structure from some information, and then going back to predict sequence and iterating.

## Masking rules during training

Typically for BERT models one masks 15% of tokens selected at random. The ESM3 authors say they found this had poor results for ESM3. They also introduce some masking methods for training that are specific to the biology under consideration in addition to for the sake of performance.

 1. The overall masking rate is higher than 15%, the mean sampling rate is 30%
    - In practice the masking rate is sampled from $\beta$(3,9) distribution 80% of the time, and from a uniform distribution the remainder of the time.
 2. For structures just dropping individual residues would likely be too easy, and one wants the model to be able to predict larger regions. To that end the ESM3 authors perform "span dropping" 50% of the time - that is dropping contiguous regions instead.
    - For the standard masking rate that occurs the other 50% of the time a cosine noise schedule is used, which means that very heavy masking of structures is common.
 3. Track Dropout. Since one wants to be able to do protein folding, in that case users would only input information from one track, namely the sequence. So, during training the model must receive only information from one track and have to predict all outputs. So the ESM3 authors mask entire tracks during training which they call dropout, although sequence is not dropped. Structure tokens are dropped 25% of the time, and structure co-ordinates are additionally dropped 50% of the time. 
    - Since entire structures are dropped somewhat often, and the masking of structure input is quite high, this means the model often has to predict a large number of structural tokens during training.
    - One the other hand, predicting sequence from structure is only done by filling masked positions and not predicting the entire sequence.

## VQVAE tokenization and structure generation
While sequence generation occurs via a typical BERT style objective of guessing tokens at masked positions, the process for structures is a little different. Firstly, how one should tokenize protein structures is not as obvious as protein sequence (which can be encoded by tokens for each residue/amino acid). Every residue, while made of the same atoms, can be arranged spatially in different ways, and we need our tokenization to take that into account.

ESM3 uses a VQVAE[^vqvae] model for both the purposes of encoding structures into tokens and generating the actual output structures from any tokens the model predicts. VQVAE models are Vector-Quantized VAEs, they behave like VAEs but have a controlled information capacity in the bottleneck that is provided through discretizing the vectors in the latent space to integers. The discretization occurs via co-learning a set of codebook vectors to which learned latent vectors are mapped to by proximity. This approach is quite powerful for learning image representations and has also been adapted into a VQGAN[^vqgan] approach which makes use of the fact that the integer tokens in the latent space can be incorporated into a transformer model.

The structure inputs of ESM3 to the VQVAE are modified into the protein backbone frames only, and additionally relative position is encoded (as per typical transformer modeling). The encoder of the VQVAE uses a geometric self-attention layer which is described in the paper, which handles translation/rotation in an invariant way, and is similar to flash attention. During the last stages of the encoding process the structural neighborhood of a residue is directly taken into account in a small layer. ultimately, each residue is assigned token. This is convenient for use with the sequence model, as with this approach both tracks have the same number of tokens as the number of residues in the protein. They use 4096 codebook tokens, so that any residue can only be mapped to one of these 4096 tokens. If one thinks about this as if each residue has a set of tokens (probably not accurate, but for back of the envelope purposes), then about 200 structural types of each residue have an encoding.

VQVAE is trained in two stages: the first attempts to reconstruct the more coarse protein backbone, and a second stage reconstructs the all atom model. Of course, in the end one wants to predict the positions of all atoms in the structure. The first stage uses 8 (standard, not geometric) transformer layers, whereas the second stage uses 30 transformer layers. In the first stage the output embeddings of the transformer model are projected into 3D space for each residue, defining the position of the backbone frame. The second stage does this too, but additionally transforms all heavy atom positions inside the frames. The two stages also have different training regimes, with stage 2 including additional augmented training data which we will cover in a later section.


### Iterative decoding
Once one has the trained VQVAE how does one generate new proteins directly from sequence? During training the multi-track model the task is only to predict missing tokens, and for the case of structure often many residues are masked or even the entire structure token track is masked. So, the final model can take a set of sequence tokens, and no structure tokens as input, and return the sequence and structure tokens it believes are correct, and the pretrained VQVAE can be used to reconstruct the structure from these predicted tokens. However, recall that the multi-track model predicts the tokens as a probability distribution, for each residue there is a certain likelihood for each token. One can think about how simply taking the most likely token in each position only may not be the optimal approach for generating diverse proteins, and is similar in concept to temperature in autoregressive models. 

There is choice in exactly how to deal with this generative modeling from the token likelihood distribution. Simply taking the most likely token, argmax decoding, is efficient but doesn't give diverse solutions. Another approach is iterative decoding. In the slowest limit iterative decoding would select the token that the model is most confident in and run the model again with this token included in the original prompt. But this is slow, $O(L^3)$ vs $O(L^2)$ complexity. To make this more efficient, one can select the top-k tokens at each step to make the decoding process faster. This is the process described in the preprint. The figure of iterative decoding in the preprint looks like a typical diffusion figure in many ways, but really it is not.

This process is conceptually similar to MVTM (masked visual token modelling) described in the maskedGIT paper[^maskGIT], though in that case they are sampling images from the tokenized latent space of a VQVAE model of images. They describe an efficient scheme for non-iterative decoding of images, and test several scheduling functions for how many tokens to select at each stage.  

## Training data
One difficulty with training protein folding models generally is that while the PDB contains many structures, the amount of data is still much below the amount of text available in datasets such as the pile. We do however have a large number of protein sequences, such as in mgnify, which contains >600M protein sequences. One way then to counteract the limited structural data is to use self-distillation: train a model to predict structure, and then predict structure of a large number of additional sequences to expand the training set. Typically one keeps the structures with good confidence. This self-distillation approach was taken in the Alphafold2[^af2] paper. For Alphafold3[^af3] they use structures predicted by Alphafold2 to extend the training set, which in their case also helps prevent hallucinations from the diffusion model they use for structure inference.

ESM3 uses a distillation approach by using predicted structures from AlphafoldDB (23TiB!) and ESMAtlas (>600M proteins) to augment the structure prediction dataset.

One additional data augmentation that was performed in ESM3 is using a inverse-folding model (predict sequence from structure) to predict 5 additional potential sequences for each protein in ESMAtlas and Alphafold-DB, and 64 additional sequences for each protein in the PDB. To ensure the model doesn't overweight the predicted sequences (which can be incorrect of course), during training the true sequence is selected 50% of the time. This sequence data augmentation, other than adding additional data, is likely helpful for providing good diversity in sequence in-filling. 

### Antibodies
Interestingly among the sequence datasets a large antibody dataset - the Observed Antibody Space - is included, which is promising for the ML design of therapeutics. It is worth noting however that most sequences in the OAS are unpaired heavy and light antibody chains, whereas antibodies are made of both and how they pair is important. No antibody folding results were included in the ESM3 preprint. In Alphafold3[^af3] several figures related to the ability to predict antibody-antigen interactions, this was not discussed in ESM3, though they do mention including antibody-antigen complexes in the training data in addition to using OAS data for sequences.

### Multimer prediction

Many proteins occur as more than one individual protein brought together in a complex (called a multimer, as opposed to monomer). In the RoseTTAfold[^rosettafold] paper the authors reasoned that even though they trained only on monomers the model should be able to predict multimers if one joined the sequences by linkers. Alphafold2[^af2] was followed by Alphafold-multimer[^afm] which made some specific changes to the training process to improve the prediction of protein multimers. For ESM3, while they mention that the context is long enough for decoding multimers they do not discuss multimer performance at any length.

## Conditional Structure Generation

An example with experimental verification shows a design of a novel GFP (Green Fluorescent Protein) protein, by fixing a small number of key residues and generating the remaining protein. One can imagine this would be useful for designing new proteins that have some key function but perhaps avoiding other limiting aspects of existing proteins. 

Interestingly for the design process they use a novel "chain of thought" process. The prompt contains the key regions of the protein sequence and structure.

1. Firstly, generating structures from the prompt and scoring them for key functional site retention but novelty from known structure.
2. then, using the generated structure as part of the prompt in a second round to obtain an improved sequence for that protein structure. 
3. Steps 1,2 are repeated many times.
4. Measure performance of final designs in the lab

In addition to the repeated joint optimization, two repeated experiments are performed, such that the best GFP design from round 1 is further optimized and a second experimental assay is performed. The designed esmGFP has only 58% sequence identity to the training set, and has brightness similar to natural GFPs.

One other generative task discussed in the preprint, though without experimental validation, is designing a variant of protein that is smaller than the original but retaining the function. In their case they take a natural trypsin (PDB 1Y3V) and modify the length to be about 67% of the original and keep the active site within an angstrom RMSD of the true active site suggesting the functionality may therefore be retained.


## Final thoughts on use cases an other projects

The ESM3 model introduces some novel methods: such as the multitrack generative approach with careful masking during training, the "chain of thought" generative approach, and the VQVAE tokenization and structure generation scheme. It will be interesting to see how this model behaves on multimers, and for therapeutic proteins, especially since antibody datasets were included in training.

# References
[^esm3]: Hayes, Tomas, et al. ["Simulating 500 million years of evolution with a language model."](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1) bioRxiv (2024)
[^bert]: Devlin, Jacob, et al. ["Bert: Pre-training of deep bidirectional transformers for language understanding." ](https://arxiv.org/abs/1810.04805) arXiv preprint arXiv:1810.04805 (2018)
[^esmfold]: Lin, Zeming, et al. ["Language models of protein sequences at the scale of evolution enable accurate structure prediction."](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1) BioRxiv (2022)
[^af3]: Abramson, Josh, et al. ["Accurate structure prediction of biomolecular interactions with AlphaFold 3."](https://www.nature.com/articles/s41586-024-07487-w) Nature (2024)
[^esmif]: Hsu, Chloe, et al. ["Learning inverse folding from millions of predicted structures."](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2) International conference on machine learning. PMLR (2022)
[^vqvae]: Van Den Oord, Aaron, and Oriol Vinyals. ["Neural discrete representation learning."](https://arxiv.org/abs/1711.00937) Advances in neural information processing systems 30 (2017)
[^vqgan]: Esser, Patrick, Robin Rombach, and Bjorn Ommer. ["Taming transformers for high-resolution image synthesis."](https://arxiv.org/abs/2012.09841) Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2021)
[^maskGIT]: Chang, Huiwen, et al. ["Maskgit: Masked generative image transformer."](https://arxiv.org/abs/2202.04200) Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)
[^af2]: Jumper, John, et al. ["Highly accurate protein structure prediction with AlphaFold."](https://www.nature.com/articles/s41586-021-03819-2) nature (2021)
[^rosettafold]: Baek, Minkyung, et al. ["Accurate prediction of protein structures and interactions using a three-track neural network."](https://www.science.org/doi/10.1126/science.abj8754) Science 373.6557 (2021)
[^afm]: Evans, Richard, et al. ["Protein complex prediction with AlphaFold-Multimer."](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2) biorxiv (2021)