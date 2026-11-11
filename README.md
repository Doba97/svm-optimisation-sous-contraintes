# **Optimisation avec contrainte : le cas du SVM (Support Vector Machine)**



> Projet de groupe (4 étudiants), Master 2 Mathématiques Appliquées et Statistique, parcours Science des Données et Intelligence Artificielle, Université Rennes 2

> Cours : Optimisation appliquée au Machine Learning et au Deep Learning

> **Rendu : 12 novembre 2026**



## **Sujet du projet**



**Thème : « Optimisation avec contrainte »** (groupes 4 étudiants).



**Consigne :** présenter un problème d'optimisation avec contrainte rencontré en Machine Learning, expliquer le rôle de la contrainte et montrer comment on peut résoudre le problème d'optimisation.



**Notre choix : le SVM.** Le SVM est un problème d'optimisation convexe quadratique sous contraintes linéaires d'inégalité, où les contraintes sont ce qui définit la notion de marge et de vecteurs de support.



## **1. Introduction générale**



### **1.1. Contexte du projet**



La machine à vecteurs de support (Support Vector Machine, SVM) est un algorithme d’apprentissage supervisé utilisé pour résoudre des problèmes de classification. Son principe consiste à déterminer une frontière de décision qui sépare les classes en maximisant la marge entre cette frontière et les observations les plus proches.



La construction de cette frontière repose sur un problème d’optimisation mathématique sous contraintes. Le SVM linéaire à marge dure cherche à minimiser la norme du vecteur de poids, tout en imposant une séparation correcte des observations.



### **1.2. Problématique**



Comment formuler et résoudre le problème d’optimisation sous contraintes associé au SVM afin de déterminer une frontière de décision optimale pour la classification binaire ?



Cette problématique conduit à étudier la fonction objectif, les contraintes d’inégalité, la convexité du problème et les conditions mathématiques permettant d’identifier une solution optimale.



### **1.3. Objectifs du projet**



L’objectif général est d’étudier le SVM comme une application concrète de l’optimisation mathématique sous contraintes.



Les objectifs spécifiques sont les suivants :



* Formuler le problème primal du SVM linéaire à marge dure sous forme mathématique.

* Analyser la fonction objectif, les contraintes et la convexité du problème.

* Étudier les multiplicateurs de Lagrange, le problème dual et les conditions de Karush-Kuhn-Tucker (KKT).

* Implémenter un SVM en Python à l’aide de `scikit-learn` et visualiser la frontière de décision ainsi que la marge.

* Évaluer l’influence du paramètre de régularisation \\(C\\) sur la frontière de décision et les performances de classification, dans le cas du SVM à marge souple.



## **2. Fondements théoriques du SVM**



### **2.1. Principe de la classification supervisée**



La classification supervisée consiste à apprendre une fonction capable d’attribuer une classe à une observation à partir d’exemples dont les classes sont connues.



Formellement, on dispose d'un échantillon d'apprentissage $S = \{(x_i, y_i)\}_{i=1}^n$, où $x_i \in \mathbb{R}^d$ est le vecteur des variables explicatives de la $i$-ème observation et $y_i \in \{-1,+1\}$ son étiquette (classification binaire). Les couples $(x_i,y_i)$ sont supposés i.i.d. selon une loi $P$ inconnue.



Un **classifieur** est une application $h : \mathbb{R}^d \to \{-1,+1\}$. Sa qualité se mesure par le **risque** $R(h) = \mathbb{P}\big(h(X) \neq Y\big)$, inconnu puisque $P$ l'est. On ne peut donc minimiser que le **risque empirique** :



$$

\widehat{R}_n(h) = \frac1n \sum{i=1}^n \mathbf{1}\{h(x_i) \neq y_i\}.

$$



On se restreint ici aux **classifieurs linéaires** :



$$

h(x) = \mathrm{sign}\big(f(x)\big), \qquad f(x) = w^\top x + b, \quad w \in \mathbb{R}^d\setminus\{0\},\ b \in \mathbb{R}.

$$



Deux difficultés motivent le SVM :



* Minimiser $\widehat R_n$ directement est un problème combinatoire non convexe, difficile en général.

* Lorsque les données sont séparables, une infinité de couples $(w,b)$ annulent $\widehat R_n$. Il faut un **critère supplémentaire** pour choisir parmi eux.



\<p align="center">

  \<img *src*="reports/figures/fig1_infinite_hyperplans.png" *alt*="Plusieurs hyperplans séparant deux classes de points" *width*="560">

\</p>

\<p align="center">\<em>Figure 1 – Lorsque les données sont séparables, il existe une infinité d’hyperplans qui les séparent.\</em>\</p>



Le SVM apporte ce critère : parmi toutes les frontières qui séparent, on retient celle qui est la plus éloignée des données.



### **2.2. Hyperplan, marge et vecteurs de support**



#### **a. Hyperplan séparateur**



Un l'hyperplan d'équation $w^\top x + b = 0$ est l'ensemble $H = \{x \in \mathbb{R}^d : w^\top x + b = 0\}$.



\<p align="center">

  \<img *src*="reports/figures/fig2_hyperplan_separateur.png" *alt*="Hyperplan séparateur de deux classes" *width*="480">

\</p>

\<p align="center">\<em>Figure 2 – Un hyperplan d’équation ⟨w, x⟩ + b = 0 sépare les deux classes.\</em>\</p>





Le vecteur $w$ est orthogonal à $H$. en effet, Si $x_1, x_2 \in H$, alors $w^\top(x_1 - x_2) = (w^\top x_1 + b) - (w^\top x_2 + b) = 0$. Donc $w$ est orthogonal à tout vecteur directeur de $H$.



Soit $d$ **la distance à l'hyperplan**,  Pour tout $x \in \mathbb{R}^d$,



$$

d(x, H) = \frac{|w^\top x + b|}{\|w\|}.

$$



*\*en effet\**, La projection orthogonale de $x$ sur $H$ est de la forme $x_p = x - t\\,w$. La condition $w^\top x_p + b = 0$ donne $t = (w^\top x + b)/\|w\|^2$. Alors $d(x,H) = \|x - x_p\| = |t|\\,\|w\| = |w^\top x + b|/\|w\|$. $\square$



Le signe de $w^\top x + b$ indique de quel côté de $H$ se trouve $x$, et sa valeur absolue mesure l'éloignement (à $\|w\|$ près).





Les données sont **linéairement séparables** s'il existe $(w,b)$ tel que $y_i(w^\top x_i + b) > 0$ pour tout $i$.



 Comme l'échantillon est fini, on pose $m = \min_i y_i(w^\top x_i + b) > 0$. En remplaçant $(w,b)$ par $(w/m, b/m)$, on obtient $y_i(w^\top x_i + b) \geq 1$ pour tout $i$, avec égalité pour au moins un point. C'est ce qui justifie la normalisation ci-dessous.



#### **b. Marge géométrique**



La **marge géométrique** de $(w,b)$ sur $S$ est



$$

\gamma(w,b) = \min{i} \frac{y_i(w^\top x_i + b)}{\|w\|}.

$$



Pour un hyperplan qui sépare, $\gamma$ est la distance de $H$ au point le plus proche. Elle est **invariante** par $(w,b) \mapsto (\lambda w, \lambda b)$ avec $\lambda>0$, ce qui est normal : c'est une propriété de l'hyperplan, pas de sa représentation.



**Normalisation canonique.** Pour lever cette indétermination d'échelle, on impose



$$

\min_i y_i(w^\top x_i + b) = 1.

$$



Alors $\gamma = 1/\|w\|$, et la **bande de séparation** délimitée par les deux hyperplans



$$

H_{+} : w^\top x + b = +1, \qquad H_{-} : w^\top x + b = -1

$$



a pour largeur



$$

\frac{2}{\|w\|}.

$$



Aucune donnée ne se trouve strictement à l'intérieur de cette bande. **Maximiser la marge revient donc à minimiser $\|w\|$.**



\<p align="center">

  \<img *src*="reports/figures/fig3_marge_maximale.png" *alt*="Hyperplan de marge maximale" *width*="620">

\</p>

\<p align="center">\<em>Figure 3 – La marge recherchée. Chaque bord de la bande est à la distance M = 1/‖w‖ de l’hyperplan, donc la largeur totale de la marge vaut 2/‖w‖.\</em>\</p>





#### **c. Vecteurs de support**



Les **vecteurs de support** sont les observations situées exactement sur le bord de la marge :



$$

y_i(w^\top x_i + b) = 1.

$$



Les autres observations vérifient $y_i(w^\top x_i + b) > 1$. Nous démontrerons à la section 4, grâce aux conditions KKT, que **la solution ne dépend que des vecteurs de support** : supprimer un point qui n'en est pas un ne modifie pas l'hyperplan optimal.



\<p align="center">

  \<img *src*="reports/figures/fig4_vecteurs_support.png" *alt*="Vecteurs de support sur le bord de la marge" *width*="520">

\</p>

\<p align="center">\<em>Figure 4 – Les vecteurs de support, situés exactement sur le bord de la marge (αᵢ\* > 0), déterminent seuls l’hyperplan optimal ⟨w\*, x⟩ + b\* = 0.\</em>\</p>





## **3. Formulation mathématique du problème d'optimisation**



Cette section transforme l'idée « choisir l'hyperplan de marge maximale » en un problème d'optimisation précis. 



#### **a. Cas separable**



Le SVM à marge dure retient, parmi les hyperplans qui séparent, celui de marge maximale :



$$

\max{w,b}\ \frac{2}{\|w\|} \quad\Longleftrightarrow\quad \min{w,b}\ \|w\| \quad\Longleftrightarrow\quad \min{w,b}\ \tfrac12\|w\|^2 .

$$



La dernière équivalence vient du fait que $t \mapsto \tfrac12 t^2$ est strictement croissante sur $\mathbb{R}_+$ : elle préserve les minimiseurs. Le facteur $\tfrac12$ n'a aucune incidence sur la solution, il simplifie seulement les dérivées.



$$

\min{w,b}\ \tfrac12\|w\|^2 \quad \text{s.c.} \quad y_i(w^\top x_i + b) \geq 1,\ \ i=1,\dots,n.

$$





Ce problème n'a de solution que si les données sont linéairement séparables : sinon, l'ensemble admissible est vide.



La marge dure est aussi très sensible aux points aberrants : une seule observation mal placée peut réduire fortement la marge, voire rendre le problème infaisable.



#### **b Cas non separable**



On autorise des violations des contraintes à l'aide de **variables d'écart** $\xi_i \geq 0$ :



$$

\min{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum{i=1}^n \xi_i \quad \text{s.c.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i,\ \ \xi_i \geq 0.

$$



**Interprétation de $\xi_i$.**



* $\xi_i = 0$ : le point est bien classé et hors de la marge (ou sur son bord).

* $0 < \xi_i < 1$ : le point est du bon côté de $H$, mais **à l'intérieur de la marge**.

* $\xi_i \geq 1$ : le point est du mauvais côté de $H$ (ou sur $H$ si $\xi_i = 1$). Il est mal classé.



\<p align="center">

  \<img *src*="reports/figures/fig5_marge_souple.png" *alt*="Cas non séparable avec marge souple" *width*="560">

\</p>

\<p align="center">\<em>Figure 5 – Cas non séparable : certains points sont dans la marge ou du mauvais côté de l’hyperplan (ξᵢ > 0).\</em>\</p>



$\sum_i \xi_i$ majore donc le nombre d'erreurs de classification, ce qui en fait un substitut convexe du risque empirique.



**Le paramètre $C > 0$ arbitre entre deux objectifs :**



* le terme $\tfrac12\|w\|^2$ favorise une marge large (régularisation) ;

* le terme $C\sum_i\xi_i$ pénalise les violations (ajustement aux données).



Si $C$ est grand, les erreurs sont fortement pénalisées : la marge se rétrécit et on tend vers la marge dure (risque de surapprentissage). Si $C$ est petit, on accepte plus d'erreurs pour obtenir une marge plus large (risque de sous-apprentissage).



### **3.1. Fonction objectif**



On note



$$

F(w,b) = \tfrac12\|w\|^2 = \tfrac12 \sum{j=1}^d w_j^2 .

$$



* $F$ est de classe $C^\infty$, avec $\nabla_w F = w$ et $\nabla^2_w F = I_d$.

* $F$ est **strictement convexe en $w$**, puisque son hessien $I_d$ est défini positif.

* $F$ ne dépend pas de $b$. Vue comme fonction de $(w,b)$, elle est convexe mais **non strictement convexe** : son hessien $\operatorname{diag}(I_d, 0)$ est seulement semi-défini positif. Cette nuance interviendra pour l'unicité de $b$ (section 4).



On pénalise $\|w\|^2$ plutôt que $\|w\|$ pour deux raisons : la différentiabilité en $w=0$, et la structure quadratique, qui conduira à un problème dual simple.



### **3.2. Contraintes d'inégalité**



Pour que l'hyperplan sépare les données et qu'aucun point ne soit dans la bande, on impose, pour chaque observation,



$$

y_i(w^\top x_i + b) \geq 1 .

$$



Pour se placer dans le cadre standard « $g_i \le 0$ » de l'optimisation sous contraintes, on pose



$$

g_i(w,b) = 1 - y_i(w^\top x_i + b) \leq 0, \qquad i=1,\dots,n .

$$



* Chaque $g_i$ est **affine** en $(w,b)$ : le coefficient de $(w,b)$ est $-y_i(x_i,1)$.

* L'ensemble admissible $\mathcal{C} = \{(w,b) : g_i(w,b)\le 0,\ \forall i\}$ est une intersection de $n$ demi-espaces fermés de $\mathbb{R}^{d+1}$, c'est-à-dire un **polyèdre convexe**.

* Cet ensemble peut être vide (données non séparables), ce qui motivera la marge souple.



**Contraintes actives et inactives.** À un point admissible $(w,b)$, la contrainte $i$ est **active** si $g_i(w,b)=0$ (le point est sur le bord de la marge) et **inactive** si $g_i(w,b)<0$. Les vecteurs de support sont donc les observations dont la contrainte est active à l'optimum. C'est le point central du rôle des contraintes : *\*seules les contraintes actives déterminent la solution\**.



### **3.4. Formulation primale du SVM**



#### **Marge dure**



$$

\boxed{

\begin{aligned}

\min{w\in\mathbb{R}^d,\ b\in\mathbb{R}}\quad & \tfrac12\|w\|^2 \\\\

\text{s.c.}\quad & y_i(w^\top x_i + b) \geq 1, \qquad i=1,\dots,n .

\end{aligned}}

$$



C'est un **programme quadratique convexe** (QP) : objectif quadratique convexe, contraintes affines. Il compte $d+1$ variables et $n$ contraintes.



**Écriture matricielle** (utile pour l'implémentation). Posons $z=(w,b)\in\mathbb{R}^{d+1}$, $P=\operatorname{diag}(I_d,0)$ et $A\in\mathbb{R}^{n\times(d+1)}$ de lignes $A_i = y_i\\,(x_i^\top,\\,1)$. Le problème s'écrit



$$

\min_z\ \tfrac12 z^\top P z \quad \text{s.c.}\quad Az \geq \mathbf{1}_n ,

$$



forme standard des solveurs de QP (`cvxopt`, `scipy.optimize` avec SLSQP).





#### **Marge souple**



$$

\boxed{

\begin{aligned}

\min{w,\\,b,\\,\xi}\quad & \tfrac12\|w\|^2 + C\sum{i=1}^n \xi_i \\\\

\text{s.c.}\quad & y_i(w^\top x_i + b) \geq 1 - \xi_i, \qquad i=1,\dots,n,\\\\

& \xi_i \geq 0, \qquad i=1,\dots,n .

\end{aligned}}

$$



C'est encore un QP convexe, avec $d+1+n$ variables et $2n$ contraintes affines. Il est **toujours admissible** : le choix $w=0$, $b=0$, $\xi_i=1$ pour tout $i$ vérifie les contraintes.



#### **Comparaison**



\| | Marge dure | Marge souple |

\|---|---|---|

\| Variables | $(w,b)$, soit $d+1$ | $(w,b,\xi)$, soit $d+1+n$ |

\| Contraintes | $n$ | $2n$ |

\| Objectif | $\tfrac12\lVert w\rVert^2$ | $\tfrac12\lVert w\rVert^2 + C\sum_i\xi_i$ |

\| Admissibilité | Seulement si séparable | Toujours |

\| Hyperparamètre | Aucun | $C>0$ |



**Existence et unicité.** Pour la marge dure (données séparables), la solution $(w^\*,b^\*)$ existe et est unique : l'unicité de $w^\*$ découle de 3.3, celle de $b^\*$ sera établie à la section 4. Pour la marge souple, le vecteur $w^\*$ est unique par stricte convexité en $w$, et $\xi^\*$ est alors déterminé par $(w^\*,b^\*)$ ; en revanche, $b^\*$ peut ne pas être unique dans des cas dégénérés.



### **À retenir**



1\. Maximiser la marge $2/\|w\|$ équivaut à minimiser $\tfrac12\|w\|^2$, fonction strictement convexe en $w$.

2\. Les contraintes $y_i(w^\top x_i+b)\ge1$ sont affines : l'ensemble admissible est un polyèdre convexe.

3\. Géométriquement, $w^\*$ est la projection de l'origine sur la projection de ce polyèdre, d'où existence et unicité de $w^\*$.

4\. Seules les contraintes actives, associées aux vecteurs de support, bloquent la solution.

5\. Le SVM est un **programme quadratique convexe**, à marge dure ou à marge souple.



## **4. Méthodes de résolution du problème d’optimisation sous contraintes**



### **4.1. Formulation du problème primal**



On considère le problème d’optimisation du SVM linéaire à marge dure :



$$

\boxed{

\begin{aligned}

\min{w,b}\quad & \frac12\|w\|^2\\\\

\text{sous contraintes}\quad & y_i(w^\top x_i+b)\geq 1, \quad i=1,\ldots,n.

\end{aligned}

}

$$



Le Lagrangien associé au problème primal, avec les multiplicateurs de Lagrange $\alpha_i\geq0$ ($i=1,\ldots,n$), est :



$$

\boxed{

L(w,b,\alpha)=

\frac12\|w\|^2+

\sum{i=1}^{n}\alpha_i

\left[1-y_i(w^\top x_i+b)\right]

}

$$



Les conditions KKT comprennent la stationnarité, la faisabilité primale, la faisabilité duale et la complémentarité.



Le système des conditions KKT s’écrit :



$$

\boxed{

\left\{

\begin{aligned}

&\nabla_w L(w,b,\alpha)=0\\\\[4pt]

&\frac{\partial L(w,b,\alpha)}{\partial b}=0\\\\[4pt]

&1-y_i(w^\top x_i+b)\leq0, \quad i=1,\ldots,n\\\\[4pt]

&\alpha_i\geq0, \quad i=1,\ldots,n\\\\[4pt]

&\alpha_i\left[1-y_i(w^\top x_i+b)\right]=0, \quad i=1,\ldots,n

\end{aligned}

\right.

}

$$



#### **a) Conditions de stationnarité**



En dérivant le Lagrangien par rapport à $w$, on obtient :



$$

\nabla_w L

\=\nabla_w\left(\frac12 w^\top w\right)

-\sum{i=1}^{n}\alpha_i y_i x_i.

$$



Or,



$$

\nabla_w\left(\frac12 w^\top w\right)=w.

$$



Par conséquent,



$$

\nabla_w L=w-\sum{i=1}^{n}\alpha_i y_i x_i.

$$



La condition $\nabla_w L=0$ équivaut donc à :



$$

\boxed{w=\sum{i=1}^{n}\alpha_i y_i x_i}

\tag{1}

$$



De même, en dérivant par rapport à $b$ :



$$

\frac{\partial L}{\partial b}

\=-\sum{i=1}^{n}\alpha_i y_i.

$$



La condition $\frac{\partial L}{\partial b}=0$ équivaut à :



$$

\boxed{\sum{i=1}^{n}\alpha_i y_i=0}

\tag{2}

$$



Les conditions de stationnarité peuvent ainsi être regroupées sous la forme :



$$

\boxed{

\left\{

\begin{aligned}

&w=\sum{i=1}^{n}\alpha_i y_i x_i\\\\[6pt]

&\sum{i=1}^{n}\alpha_i y_i=0

\end{aligned}

\right.

}

$$



#### **b) Conditions de faisabilité et de complémentarité**



Les contraintes primales et duales restent :



$$

\boxed{

\left\{

\begin{aligned}

&y_i(w^\top x_i+b)\geq1, \quad i=1,\ldots,n\\\\[4pt]

&\alpha_i\geq0, \quad i=1,\ldots,n

\end{aligned}

\right.

}

$$



La complémentarité s’écrit :



$$

\alpha_i\left[1-y_i(w^\top x_i+b)\right]=0.

$$



Elle équivaut à la disjonction suivante :



$$

\boxed{

\alpha_i=0

\quad\text{ou}\quad

y_i(w^\top x_i+b)=1

}

$$



pour chaque observation $i$. Ainsi, lorsqu’un multiplicateur est strictement positif, l’observation correspondante est située sur la marge et constitue un vecteur de support.



#### **c) Déduction de la formulation duale**



À partir de la première condition de stationnarité,



$$

w=\sum{i=1}^{n}\alpha_i y_i x_i,

$$



on remplace $w$ dans le Lagrangien. En utilisant également la deuxième condition,



$$

\sum{i=1}^{n}\alpha_i y_i=0,

$$



on obtient :



$$

\begin{aligned}

L(\alpha)

&=\frac12\|w\|^2+\sum{i=1}^{n}\alpha_i

-\sum{i=1}^{n}\alpha_i y_iw^\top x_i

-b\sum{i=1}^{n}\alpha_i y_i\\\\

&=\sum{i=1}^{n}\alpha_i-\frac12\|w\|^2.

\end{aligned}

$$



Or,



$$

\|w\|^2=

\sum{i=1}^{n}\sum{j=1}^{n}

\alpha_i\alpha_j y_i y_j x_i^\top x_j.

$$



On en déduit le problème dual :



$$

\boxed{

\begin{cases}

\displaystyle\max{\alpha}\quad

\sum{i=1}^{n}\alpha_i

-\frac12\sum{i=1}^{n}\sum{j=1}^{n}

\alpha_i\alpha_jy_iy_jx_i^\top x_j\\\\[10pt]

\displaystyle\sum{i=1}^{n}\alpha_i y_i=0\\\\[6pt]

\alpha_i\geq0,\quad i=1,\ldots,n

\end{cases}

}

$$



La résolution de ce problème permet de déterminer les multiplicateurs optimaux $\alpha_i^\*$. On calcule ensuite :



$$

\boxed{w^\*=\sum{i=1}^{n}\alpha_i^\*y_ix_i}

$$



Puis, pour un vecteur de support $x_k$ tel que $\alpha_k^\*>0$, la condition de complémentarité donne :



$$

y_k(w^{\*\top}x_k+b^\*)=1.

$$



On multiplie les deux côtés par $y_k$. Comme $y_k\in\{-1,+1\}$, on a $y_k^2=1$, donc



$$

w^{\*\top}x_k+b^\*=y_k\quad\Longrightarrow\quad b^\*=y_k-w^{\*\top}x_k .

$$



L’hyperplan optimal est donc défini par $w^{\*\top}x+b^\*=0$.



### **4.2. Cas non séparable : SVM à marge souple**



Lorsque les données ne sont pas linéairement séparables, le problème de la section 4.1 n’a pas de solution. On introduit des variables d’écart $\xi_i\geq0$ et on suit exactement la même démarche.



#### **a) Formulation du problème primal**



$$

\boxed{

\begin{aligned}

\min{w,b,\xi}\quad & \frac12\|w\|^2+C\sum{i=1}^{n}\xi_i\\\\

\text{sous contraintes}\quad & y_i(w^\top x_i+b)\geq 1-\xi_i, \quad i=1,\ldots,n,\\\\

& \xi_i\geq0, \quad i=1,\ldots,n.

\end{aligned}

}

$$



On réécrit les contraintes sous la forme :



$$

g_i(w,b,\xi)=1-\xi_i-y_i(w^\top x_i+b)\leq 0,

\qquad

h_i(\xi)=-\xi_i\leq 0 .

$$



On introduit les multiplicateurs de Lagrange $\alpha_i\geq0$ (pour les contraintes $g_i\leq0$) et $\mu_i\geq0$ (pour les contraintes $h_i\leq0$), pour $i=1,\ldots,n$.



#### **b) Construction du Lagrangien**



$$

\boxed{

L(w,b,\xi,\alpha,\mu)=

\frac12\|w\|^2+C\sum{i=1}^{n}\xi_i

+\sum{i=1}^{n}\alpha_i\left[1-\xi_i-y_i(w^\top x_i+b)\right]

-\sum{i=1}^{n}\mu_i\\,\xi_i

}

$$



#### **c) Établissement des conditions KKT**



Le système des conditions KKT s’écrit :



$$

\boxed{

\left\{

\begin{aligned}

&\nabla_w L=0,\qquad \frac{\partial L}{\partial b}=0,\qquad \frac{\partial L}{\partial \xi_i}=0\\\\[4pt]

&1-\xi_i-y_i(w^\top x_i+b)\leq0,\quad \xi_i\geq0, \quad i=1,\ldots,n\\\\[4pt]

&\alpha_i\geq0,\quad \mu_i\geq0, \quad i=1,\ldots,n\\\\[4pt]

&\alpha_i\left[1-\xi_i-y_i(w^\top x_i+b)\right]=0, \quad i=1,\ldots,n\\\\[4pt]

&\mu_i\\,\xi_i=0, \quad i=1,\ldots,n

\end{aligned}

\right.

}

$$



**Conditions de stationnarité.** En dérivant le Lagrangien :



$$

\nabla_w L=w-\sum{i=1}^{n}\alpha_i y_i x_i,

\qquad

\frac{\partial L}{\partial b}=-\sum{i=1}^{n}\alpha_i y_i,

\qquad

\frac{\partial L}{\partial \xi_i}=C-\alpha_i-\mu_i .

$$



Elles équivalent donc à :



$$

\boxed{

\left\{

\begin{aligned}

&w=\sum{i=1}^{n}\alpha_i y_i x_i &&(3)\\\\[6pt]

&\sum{i=1}^{n}\alpha_i y_i=0 &&(4)\\\\[6pt]

&C-\alpha_i-\mu_i=0, \quad i=1,\ldots,n &&(5)

\end{aligned}

\right.

}

$$



Les conditions (3) et (4) sont celles du cas séparable. La condition (5) est nouvelle : comme $\mu_i\geq0$, elle donne $\mu_i=C-\alpha_i\geq0$, c’est-à-dire



$$

\boxed{0\leq\alpha_i\leq C,\quad i=1,\ldots,n.}

$$



**Conditions de complémentarité.** Elles s’écrivent



$$

\alpha_i=0\quad\text{ou}\quad y_i(w^\top x_i+b)=1-\xi_i,

\qquad\qquad

\mu_i\\,\xi_i=0\iff(C-\alpha_i)\\,\xi_i=0\iff\alpha_i=C\quad\text{ou}\quad\xi_i=0 .

$$



On en déduit trois situations pour une observation $i$, notée $f(x_i)=w^\top x_i+b$ :



\| Valeur de $\alpha_i$ | Conséquence | Position du point |

\|---|---|---|

\| $\alpha_i=0$ | $\mu_i=C>0$, donc $\xi_i=0$, et $y_if(x_i)\geq1$ | hors de la marge ou sur son bord, **bien classé** |

\| $0<\alpha_i\<C$ | $\xi_i=0$ et $y_if(x_i)=1$ | **exactement sur le bord** de la marge (vecteur de support « libre ») |

\| $\alpha_i=C$ | $y_if(x_i)=1-\xi_i\leq1$ | **dans la marge ou mal classé** (vecteur de support « saturé ») |



Le paramètre $C$ plafonne donc l’influence d’une observation : plus $C$ est petit, plus il y a d’observations pour lesquelles $\alpha_i=C$.



#### **d) Déduction de la formulation duale**



On remplace $w$ par $\sum_i\alpha_iy_ix_i$ dans le Lagrangien, et on utilise (4) :



$$

\begin{aligned}

L

&=\frac12\|w\|^2+\sum{i=1}^{n}\alpha_i

-\sum{i=1}^{n}\alpha_i y_iw^\top x_i

-b\sum{i=1}^{n}\alpha_i y_i

+\sum{i=1}^{n}\left(C-\alpha_i-\mu_i\right)\xi_i\\\\

&=\sum{i=1}^{n}\alpha_i-\frac12\|w\|^2 ,

\end{aligned}

$$



car $\sum_i\alpha_iy_iw^\top x_i=\|w\|^2$, $\sum_i\alpha_iy_i=0$ et, d’après (5), le dernier terme est nul : les variables $\xi_i$ et $\mu_i$ disparaissent. Avec $\|w\|^2=\sum{i,j}\alpha_i\alpha_jy_iy_jx_i^\top x_j$ et la contrainte $0\leq\alpha_i\leq C$ issue de (5), on obtient le problème dual :



$$

\boxed{

\begin{cases}

\displaystyle\max{\alpha}\quad

\sum{i=1}^{n}\alpha_i

-\frac12\sum{i=1}^{n}\sum{j=1}^{n}

\alpha_i\alpha_jy_iy_jx_i^\top x_j\\\\[10pt]

\displaystyle\sum{i=1}^{n}\alpha_i y_i=0\\\\[6pt]

0\leq\alpha_i\leq C,\quad i=1,\ldots,n

\end{cases}

}

$$



**Comparaison avec le cas séparable.** La fonction objectif est la même ; **seule la borne supérieure $\alpha_i\leq C$ est ajoutée**. Le cas séparable correspond à $C=+\infty$. Ce dual est toujours admissible ($\alpha=0$).



**Reconstruction de la solution.** Une fois $\alpha^\*$ déterminé :



$$

\boxed{w^\*=\sum{i=1}^{n}\alpha_i^\*y_ix_i}

$$



Pour un vecteur de support libre $x_k$, c’est-à-dire tel que $0<\alpha_k^\*\<C$, on a $\xi_k^\*=0$ et $y_k(w^{\*\top}x_k+b^\*)=1$. Comme $y_k^2=1$ :



$$

\boxed{b^\*=y_k-w^{\*\top}x_k}

$$



(en pratique, on moyenne cette valeur sur tous les vecteurs libres). Les écarts se déduisent de $\xi_i^\*=\max\big(0,\ 1-y_i(w^{\*\top}x_i+b^\*)\big)$, et la règle de décision est $f(x)=\sum_i\alpha_i^\*y_i\\,x_i^\top x+b^\*$.



## **Équipe**



\| Nom | Rôle principal | GitHub |

\|---|---|---|

\| SORO Doba Issiaka | Théorie : fondements et formulation (sections 2 et 3) | @ |

\| Ming Wei Ang | Théorie : Lagrangien, KKT, dual (section 4) | @ |

|Matis Boudelier | Code : implémentation et visualisations (section 5) | @ |

\| Alexandre Hayoun | Expériences et évaluation (section 6) | @ |





## **Structure du dépôt**



```

svm-optimisation-sous-contraintes/

├── README.md

├── requirements.txt

├── .gitignore

├── data/                # données utilisées

├── notebooks/           # exploration, théorie, expérimentations

├── src/                 # code réutilisable (data.py, svm_primal.py, plots.py)

├── tests/               # tests du code

├── reports/

│   ├── figures/         # graphiques et résultats

│   └── rapport_final    # rapport

└── presentation/        # support de présentation finale

```



## **Installation et exécution**



```bash

git clone https\://github.com/**<**organisation**>**/svm-optimisation-sous-contraintes.git

cd svm-optimisation-sous-contraintes

python -m venv .venv

source .venv/bin/activate        *# Windows : .venv\Scripts\activate*

pip install -r requirements.txt

jupyter notebook

```



Ordre conseillé d'exécution des notebooks : `01` puis `02`, `03`, `04`.



## **Reproductibilité**



- Graine aléatoire fixée dans tous les notebooks : `random_state = 42` *_(à confirmer en équipe)_*.

- Versions des dépendances figées dans `requirements.txt`.
