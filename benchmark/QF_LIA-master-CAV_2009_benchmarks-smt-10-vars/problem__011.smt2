(set-info :smt-lib-version 2.6)
(set-logic QF_LIA)
(set-info :source |
Alberto Griggio

|)
(set-info :category "random")
(set-info :status unsat)
(declare-fun x0 () Int)
(declare-fun x1 () Int)
(declare-fun x2 () Int)
(declare-fun x3 () Int)
(declare-fun x4 () Int)
(declare-fun x5 () Int)
(declare-fun x6 () Int)
(declare-fun x7 () Int)
(declare-fun x8 () Int)
(declare-fun x9 () Int)
(assert (let ((?v_10 (* 0 x4)) (?v_2 (* 1 x8)) (?v_0 (* 1 x0)) (?v_6 (* 0 x5)) (?v_4 (* 1 x1)) (?v_5 (* 0 x6)) (?v_11 (* 0 x9)) (?v_8 (* 1 x4)) (?v_9 (* 0 x2)) (?v_16 (* 0 x1)) (?v_13 (* 1 x2)) (?v_14 (* 0 x7)) (?v_23 (* 1 x9)) (?v_22 (* 1 x3)) (?v_15 (* 0 x3)) (?v_24 (* 1 x7)) (?v_12 (* 0 x8)) (?v_18 (* 0 x0)) (?v_1 (* (- 1) x0)) (?v_3 (* (- 1) x8)) (?v_7 (* (- 1) x1)) (?v_17 (* (- 1) x7)) (?v_20 (* (- 1) x3)) (?v_25 (* (- 1) x6)) (?v_19 (* (- 1) x5)) (?v_21 (* (- 1) x4))) (and (<= (+ ?v_10 ?v_2 ?v_14 ?v_0 ?v_0 ?v_1 ?v_12 ?v_1 ?v_3 ?v_5) (- 1)) (<= (+ ?v_2 ?v_2 ?v_6 ?v_4 ?v_1 ?v_3 ?v_1 ?v_7 ?v_4 ?v_5) 0) (<= (+ ?v_11 ?v_9 ?v_5 ?v_8 ?v_17 ?v_6 ?v_5 ?v_20 ?v_7 ?v_25) 0) (<= (+ ?v_8 ?v_2 ?v_9 ?v_10 ?v_19 ?v_15 ?v_11 ?v_7 ?v_16 ?v_12) 0) (<= (+ ?v_13 ?v_0 ?v_13 ?v_1 ?v_14 ?v_15 ?v_1 ?v_21 ?v_18 ?v_5) (- 1)) (<= (+ ?v_4 ?v_16 ?v_11 ?v_17 ?v_6 ?v_23 ?v_1 ?v_22 ?v_1 ?v_6) 0) (<= (+ ?v_4 ?v_2 ?v_11 ?v_8 ?v_5 ?v_18 ?v_5 (* (- 1) x2) ?v_0 ?v_10) 1) (<= (+ ?v_15 ?v_10 ?v_19 ?v_24 ?v_9 ?v_10 ?v_5 ?v_16 ?v_20 ?v_3) 0) (<= (+ ?v_12 ?v_14 ?v_12 ?v_8 ?v_16 ?v_12 ?v_5 ?v_12 ?v_2 ?v_21) (- 1)) (<= (+ ?v_0 ?v_10 ?v_14 ?v_16 ?v_9 ?v_9 ?v_6 ?v_6 ?v_5 ?v_5) 1) (<= (+ ?v_13 ?v_22 ?v_17 ?v_18 ?v_12 ?v_20 ?v_9 ?v_16 ?v_12 ?v_11) 0) (<= (+ ?v_11 ?v_20 ?v_2 ?v_0 ?v_23 ?v_21 ?v_22 ?v_16 ?v_18 ?v_16) (- 1)) (<= (+ ?v_18 ?v_24 ?v_3 ?v_14 ?v_0 ?v_20 ?v_10 ?v_14 ?v_11 ?v_19) 0) (<= (+ ?v_6 ?v_24 ?v_12 ?v_20 ?v_8 ?v_10 ?v_15 ?v_24 ?v_16 ?v_2) 0) (<= (+ ?v_8 ?v_10 ?v_18 ?v_25 ?v_14 ?v_10 ?v_0 (* 1 x5) ?v_10 ?v_18) 0))))
(check-sat)
(exit)
