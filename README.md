# From claim counts to interarrival times using a small neural framework 

Claim counting plays a central role in actuarial modelling, with distributions such as Poisson, Negative Binomial, and their zero-inflated variants commonly used in Generalised Linear Models (GLMs). Recently, there has been a growing interest in integrating traditional models with neural network approaches.

In the context of assumption-free machine learning, it is natural to incorporate the precise occurence dates of claims, using this additional data to model claim interarrival times rather than just aggregated counts. To this end, we propose a small neural framework designed to estimate the cumulative distribution of continuous time-to-event data. This framework enables the direct derivation of probability density and hazard functions, with likelihood maximisation that accounts for left-, right-, and interval censoring. Furthermore, the model can handle multiple competing risks, such as different claim types (e.g. total loss and own fault).

We demonstrate the practical application of the framework through a synthetic example. Despite variations in underlying processes, the neural network is capable of accurately reproducing claim interarrival times and counts for different sub-populations.

In conclusion, the proposed framework makes a promising contribution to the accurate and insightful modelling of claim-generating processes. By providing a more granular connection to policy terms and conditions, it has the potential to enhance pricing. Additionally, this framework offers broader applicability as a versatile tool for time-to-event analysis.
