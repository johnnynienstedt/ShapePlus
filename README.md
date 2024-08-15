# ShapePlus
Evaluation of pitch shapes - first step toward full pitcher evaluation.

The specific goal of this project is to build a machine learning algorithm which grades pitches solely by their "shape" - that is, their physical characteristics. The overarching goal of this research is to develop a method of evaluating pitchers whollistically, which will hopefully include considerations not only of pitch shape, but also inter-repertoire interactions, command/control, and biomechanical efficiency.


The following is a more in-depth explanation of this project.

We know that pitchers really control only a few aspects of their performance to a predictable degree -- namely, whiffs and the ratio of ground balls to fly balls -- so this project aims to analyze the qualities of a pitch's shape which help (or hurt) its ability to generate those outcomes. Over the span of 2021-2024, the average run values of those outcomes for the pitcher (via Baseball Savant) were:

  Swinging strike:  0.116
  Ground ball:      0.058
  Air ball:        -0.141

Using the pybaseball database, every pitch from 2021-2023 was assigned a binary value for each of these outcomes. Then, the pitches were fed into an XGBoost model, with the input variables as below:

  release_speed
  pfx_x                   (horizontal movement)
  pfx_z                   (induced vertical movement)
  release_extension
  VAA                     (calculated vertical approach angle)
  HAA                     (calculated horizontal approach angle)
  sharpness               (a metric of my own invention with units degrees of break per second)

These variables describe only the pitch's trajectory through the air; location, deception, and effects of previous pitches are intentionally ignored to avoid obfuscating the effects of pitch shape. These effects will each be the subject of other models, which together will form a hollistic pitcher evaluation tool.

The only metric I debated not including was extension, since it is more of a pither quality than a pitch quality. However, it does significantly effect the shape of the pitch (especially its percieved velocity) so I left it in.

In its current state, the model is trained separately for each pitch type; that is, there are actually several independent models, one for each pitch type. I did this to reduce the complexity of each model. Ideally, for a single pitch type, effectiveness increases monotonically with most variables, so the model will not need to use especially deep trees to make accurate predictions. 

This required a separate mini-project to reclassify pitches, since MLB's pitch classification system leaves a lot to be desired. In my opinion, it is foolish to allow pitchers to classify their own pitches, because it can confuse fans and mislead analysis. Therefore, I created a system to classify each pitcher's pitches based purely on shape. However, this classification process DOES include full-repertoire considerations, because whether an 88mph breaking ball with +1" iVB and -3" HB is a cutter or a slider or a curveball depends on if your fastball is 90mph or 95mph or 100mph.

I was not happy to do this, since I was trying to preserve repertoire independence, but there's really no way around it; if Emmanuel Clase's 92mph, +1" iVB, -7" HB pitch is called a cutter, what is his 100/+10/-3 pitch supposed to be called? Or, if we do call them a slider and a cutter, what is Calyton Kershaw's 87/+7/-5 suppsoed to be? Unfortunately, there's simply no getting around it. Fortunately, the pitch names have no bearing on the grading system. I got a bit more creative with names so that it is more immediately obvious to the reader what types of pitches a pitcher truly throws. The pitch names are:

  Riding Fastball
  Fastball
  Sinker
  Cutter
  Gyro Slider
  Two-Plane Slider
  Sweeper
  Slurve
  Curveball
  Slow Curve
  Movement-Based Changeup
  Velo-Based Changeup
  Knuckleball

Now back to the model. For each pitch type, the model considers every example of that pitch thrown from 2021-2023. It knows the shape of the pitch, it knows the platoon matchup, and it knows which (if any) of the three main outcomes it resulted in. Using that data, it builds an idea of which shapes are more likely to result in whiffs, grounders, or fly balls.

Once the model has been optimized as much as possible, it is then used to grade each pitch thrown in 2024. Based on the shape of the pitch, (and the platoon matchup) the model returns three percentages: the odds of a whiff, a ground ball, and a fly ball. These percentages don't add up to 100%, but that's not an issue; the other outcomes, such as called strike, ball, foul, etc. are not of interest to us. Since we know the run value of each outcome, we can use these percentages to calculate an expected run value for the pitch.

Next, the pitches are grouped and aggregated by pitcher and pitch type, so that each pitcher has one entry for each pitch type, with an average expected run value for that pitch type. The run values are then normalized to a mean of 100 and a standard deviation of 10. For every 100 pitcher-pitch types, we would expect to see only a few graded at 120 or above, and only a few graded at 80 or below. In analogy to the traditional 20-80 scale, a 110 Shape+ corresponds to a 60, or 'plus' pitch, and a 120 corresponds to a 70, or 'plus plus' pitch.

Model Analysis:

One of the first things I did was evaluate all pitch-types, to see if the model had any strong preferances. Unsurprisingly, it most certainly does. Since the model is based entirely off of getting whiffs and preventing fly balls, changeups are very highly rated. Gyro sliders and two-plane sliders, which have less horizontal break than sweepers, are also highly rated, mostly for their ability to generate whiffs. Bigger breaking balls tend to generate a good amount of whiffs but also surrender fly balls, so they rate out as average to below average. The big surprise (to me, at least) was that riding fastballs rated out the worst of all pithces. I intentionally separated 'riding fastballs' from just 'fastballs' to evaluate the effect of iVB on a fastball, which is generally accepted to be quite significant. However, the high iVB fastballs performed even worse on average than their low iVB counterparts. While they do generate more whiffs, the higher iVB also leads to more fly balls which significantly drag down the expected run value. Sinkers are on the opposite end of the spectrum, inducing the fewest whiffs of any pitch type, but also fairly few fly balls. Still, they rank below everything except the other fastballs.

Originally I had not separated platoon matchups in the training of the model, so after I did, I re-checked the above analysis broken down by handedness (same or different). This time, the big surprise was that changeups - both the velo-based and movement-based variety - have beeen more effective against same-handed batters than against opposite-handed batters. This is astonishing to me. Pitchers overwhelmingly throw their changeups to opposite-handed batters (18.6%) compared to same-handed batters (7.2%). However, the data is clear: changeups generate more whiffs (34% vs. 30%) AND more ground balls when thrown to same-handed batters as compared to opposite-handed batters. Now, there are many reasons why pitchers might still prefer to throw changups in platoon disadvantage situations. First, the changeup is still a very effective pitch to opposite-handed batters, whereas most breaking balls lose a great amount of effectiveness; it might be less of a preference of the changeup so much as shying away from breaking balls. Also, the success of the same-handed changeup may be partially due to its rarity; it's not a pitch that batters are used to seeing from a same-handed pitcher, so it makes sense that they would have less success against it. Still, I would not be surprised if a significant portion of this incongruity could be explained simply from the traditional notion that righty-righty changeups are ineffective. It will be interesting to see if this trend changes over time.

My next step is to run correlations of Shape+ to ERA, FIP, and other such metrics, and compare them to other similar pitch grading models. Remember, the goal is not to create a hugely predictive model on its own; the goal is to create a model that does its one job (grading pitch shapes) very well. Combining it will grades for command, repertoire, and deception will hopefully create a truly precdictive model.

According to the current iteration of the model, the individual pitches with the best shape are:
  - To same-handed hitters: German Marquez's gyro slider with a 137 Shape+, or 3.7 standard deviations above the mean.
  - To opposite-handed hitters: Luke Jackson's gyro slider with a 133 Shape+, or 3.3 standard deviations above the mean.

And the pitches with the worst shape are:
  - To same-handed hitters: Miguel Diaz's riding fastball with a 69 Shape+, or 3.1 standard deviations below the mean.
  - To opposite-handed hitters: Julio Teheran's twwo-plane slider with a 76 Shape+, or 2.4 standard deviations below the mean.

As for pitchers as a whole, the owners of highest grading repertoires are:
  - To same-handed hitters: Devin Williams, 122 Shape+
  - To opposite-handed hitters: Luke Jackson, 118 Shape+

And the lowest grading repertoires:
  - To same-handed hitters: Zach Plesac, 84 Shape+
  - To opposite-handed hitters: Kolby Allard, 89 Shape+

We can already see two interesting trends here. First, the grading of a pitcher's repertoire is much more conservative than the grading of indivisual pitches. This is because the scaling process occurs at the pitcher-pitch level. So, when looking at Devin Williams' changeup and its 131 Shape+ vs righties, we can say that it grades out roughly three standard deviations above the 'average pitch.' When we look at his entire repertoire, we see that he has a 122 Shape+ vs righites, so we might be tempted to say that his repertoire grades out as 2.2 standard deviations above the 'average pitcher.' However, this is false; since no pitcher throws his best pitch (or his worst one) every single time, the grades tend to 'compress' towards 100 for full pithcer repertoires. I am debating whether to re-scale on the pitcher-level, to resolve this issue.

The second notable trend is that the best pitches and pitchers are rated more positively than the worst are rated negatively. The two best pitches are rated an average of 3.5 standard deviations above the mean, while the two worst are rated an average of just 2.75 standard deviations below the mean. This implies that we have a positive skew to the data - that is, that the median Shape+ is lower than the mean, and that the mean is dragged upwards due to some truly exceptional pitches. In fact, for both same- and opposite-handed hitters, the mean Shape+ is (obviously) 100, while the median is a tick lower at 99. 



