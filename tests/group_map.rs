#[cfg(test)]
mod tests {
    use cycle_map::GroupMap;
    use hashbrown::HashSet;

    #[derive(PartialEq, Eq, Clone, Hash, Debug)]
    struct TestingStruct {
        pub(crate) value: u64,
        pub(crate) data: String,
    }

    impl TestingStruct {
        pub(crate) fn from_value(value: u64) -> Self {
            Self {
                value,
                data: value.to_string(),
            }
        }
    }

    fn construct_default_map() -> GroupMap<String, TestingStruct> {
        (0..10)
            .map(|i| (i.to_string(), TestingStruct::from_value(i)))
            .collect()
    }

    #[test]
    fn group_map_construction_test() {
        let map: GroupMap<String, TestingStruct> = GroupMap::new();
        assert_eq!(map.len_left(), 0);
        assert_eq!(map.len_right(), 0);
        assert_eq!(map.capacity_left(), 0);
        assert_eq!(map.capacity_right(), 0);
        let mut map = construct_default_map();
        assert_eq!(map.len_left(), 10);
        assert_eq!(map.len_right(), 10);
        let l_cap = map.capacity_left();
        let r_cap = map.capacity_right();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len_left(), 0);
        assert_eq!(map.len_right(), 0);
        assert_eq!(map.capacity_left(), l_cap);
        assert_eq!(map.capacity_right(), r_cap);
    }

    #[test]
    fn group_map_insert_tests() {
        let vals = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let mut map: GroupMap<String, usize> = GroupMap::new();

        // Simple insert
        (0..10).for_each(|i| {
            map.insert(i.to_string(), i);
            assert!(map.are_paired(&i.to_string(), &i));
        });

        // Should be the same as calling insert_right
        (0..10).for_each(|i| {
            map.insert(vals[i].to_string(), i);
            assert!(map.are_paired(&i.to_string(), &i));
            assert!(map.are_paired(&vals[i].to_string(), &i));
        });

        // Repair existing left items with new right items
        (0..10).for_each(|i| {
            map.insert(i.to_string(), 10 * i);
            assert!(!map.are_paired(&i.to_string(), &i));
            assert!(map.contains_right(&i));
        });

        // Pair existing left and right items
        for i in 1..10 {
            map.insert(i.to_string(), i);
            assert!(map.are_paired(&i.to_string(), &i));
            assert!(!map.are_paired(&i.to_string(), &(10 * i)));
            assert!(map.contains_right(&(10 * i)));
        }
    }

    #[test]
    fn group_map_insert_remove_test() {
        let mut map: GroupMap<u64, String> = GroupMap::with_capacity(100);
        for i in 0..100 {
            let opt = map.insert_remove(i, i.to_string());
            assert_eq!(opt, (None, None));
        }
        assert_eq!(map.len_left(), 100);
        assert_eq!(map.len_right(), 100);
        for (val, s) in map.iter_paired() {
            assert_eq!(val.to_string(), *s);
            assert_eq!(str::parse::<u64>(s).expect("Unreachable"), *val);
            println!("{val}, {s}");
        }
    }

    #[test]
    fn group_map_swap_right_remove_tests() {
        let mut map: GroupMap<String, TestingStruct> = construct_default_map();

        // Should be equivalent to insert_right
        for i in 20..30 {
            let opt = map.swap_right_remove(
                &TestingStruct::from_value(i - 10),
                TestingStruct::from_value(i),
            );
            assert!(opt.is_none());
            assert!(map.contains_right(&TestingStruct::from_value(i)));
            assert!(!map.is_right_paired(&TestingStruct::from_value(i)));
        }

        // Actually swap values
        for i in 10..20 {
            let opt = map.swap_right_remove(
                &TestingStruct::from_value(i - 10),
                TestingStruct::from_value(i),
            );
            assert_eq!(opt, Some(TestingStruct::from_value(i - 10)));
            assert!(map.are_paired(&(i - 10).to_string(), &TestingStruct::from_value(i)));
            assert!(!map.contains_right(&TestingStruct::from_value(i - 10)));
            assert!(!map.is_right_paired(&TestingStruct::from_value(i - 10)));
        }
    }

    #[test]
    fn group_map_swap_right_tests() {
        let mut map: GroupMap<String, TestingStruct> = construct_default_map();

        // Should be equivalent to insert_right
        for i in 20..30 {
            map.swap_right(
                &TestingStruct::from_value(i - 10),
                TestingStruct::from_value(i),
            );
            assert!(map.contains_right(&TestingStruct::from_value(i)));
            assert!(!map.is_right_paired(&TestingStruct::from_value(i)));
        }

        // Actually swap values
        for i in 10..20 {
            map.swap_right(
                &TestingStruct::from_value(i - 10),
                TestingStruct::from_value(i),
            );
            assert!(map.are_paired(&(i - 10).to_string(), &TestingStruct::from_value(i)));
            assert!(map.contains_right(&TestingStruct::from_value(i - 10)));
            assert!(!map.is_right_paired(&TestingStruct::from_value(i - 10)));
        }
    }

    #[test]
    fn group_map_get_tests() {
        let map: GroupMap<String, TestingStruct> = construct_default_map();
        assert!(map.contains_left(&0.to_string()));
        assert!(map.contains_right(&TestingStruct::from_value(0)));
        let opt = map.get_left_iter(&TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.get_left_iter(&TestingStruct::from_value(0));
        assert!(opt.is_some());
        assert_eq!(Some(&"0".to_string()), opt.unwrap().next());
        let opt = map.get_right(&"42".to_string());
        assert!(opt.is_none());
        let opt = map.get_right(&"0".to_string());
        assert_eq!(opt, Some(&TestingStruct::from_value(0)));
    }

    #[test]
    fn group_map_remove_tests() {
        // Double remove
        let mut map: GroupMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove(&"42".to_string(), &TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.remove(&"0".to_string(), &TestingStruct::from_value(0));
        assert_eq!(
            opt,
            Some((
                vec!["0".to_string()].into_iter().collect(),
                TestingStruct::from_value(0)
            ))
        );
        // Left remove
        let mut map: GroupMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove_right(&TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.remove_right(&TestingStruct::from_value(0));
        assert_eq!(
            opt,
            Some((
                vec!["0".to_string()].into_iter().collect(),
                TestingStruct::from_value(0)
            ))
        );
        // Right remove
        let mut map: GroupMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove_left(&"42".to_string());
        assert!(opt.is_none());
        let opt = map.remove_left(&"0".to_string());
        assert_eq!(opt, Some("0".to_string()));
    }

    #[test]
    fn group_map_retain_paired() {
        let mut map: GroupMap<u64, String> = GroupMap::with_capacity(100);
        for i in 0..100 {
            if i < 50 {
                let opt = map.insert_remove(i, i.to_string());
                assert_eq!(opt, (None, None));
            } else {
                let opt = map.insert_right(i.to_string());
                assert_eq!(opt, None);
            }
        }
        assert_eq!(map.len_left(), 50);
        assert_eq!(map.len_right(), 100);
        map.retain_paired(|l, _| *l % 2 == 0);
        assert_eq!(map.len_left(), 25);
        assert_eq!(map.len_right(), 100);
        for op in map.iter() {
            println!("{op:?}");
            if let Some(x) = op.0 {
                assert_eq!(x % 2, 1);
            }
        }
    }

    #[test]
    fn group_map_retain_unpaired() {
        let mut map: GroupMap<u64, String> = GroupMap::with_capacity(100);
        for i in 0..100 {
            if i < 50 {
                let opt = map.insert_remove(i, i.to_string());
                assert_eq!(opt, (None, None));
            } else {
                let opt = map.insert_right(i.to_string());
                assert_eq!(opt, None);
            }
        }
        assert_eq!(map.len_left(), 50);
        assert_eq!(map.len_right(), 100);
        map.retain_unpaired(|val| val.parse::<u64>().unwrap() % 2 == 0);
        assert_eq!(map.len_left(), 50);
        assert_eq!(map.len_right(), 75);
    }

    #[test]
    fn group_map_iter_tests() {
        // Main iter
        let map = construct_default_map();
        let iter = map.iter();
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        // Left iter
        let map = construct_default_map();
        let iter = map.iter_left();
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        assert_eq!(
            iter.cloned().collect::<HashSet<String>>(),
            (0..10).map(|i| i.to_string()).collect::<HashSet<String>>()
        );
        // Right iter
        let map = construct_default_map();
        let iter = map.iter_right();
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        assert_eq!(
            iter.cloned().collect::<HashSet<TestingStruct>>(),
            (0..10)
                .map(TestingStruct::from_value)
                .collect::<HashSet<TestingStruct>>()
        );
    }

    #[test]
    fn group_map_paired_iter_tests() {
        // paired iter
        let map = construct_default_map();
        let iter = map.iter_paired();
        println!("{map:?}");
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        assert_eq!(
            iter.map(|(l, r)| (l.clone(), r.clone()))
                .collect::<GroupMap<String, TestingStruct>>(),
            map
        );
        let map: GroupMap<String, TestingStruct> = (0..10)
            .map(|i| (None, TestingStruct::from_value(i)))
            .collect();
        let iter = map.iter_paired();
        println!("{iter:?}");
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.clone().len(), 0);
        assert_eq!(
            iter.map(|(l, r)| (l.clone(), r.clone()))
                .collect::<GroupMap<String, TestingStruct>>(),
            GroupMap::new()
        );
    }

    #[test]
    fn group_map_unpaired_iter_tests() {
        // Unpaired iter
        let map: GroupMap<String, TestingStruct> = (0..10)
            .map(|i| (None, TestingStruct::from_value(i)))
            .collect();
        let iter = map.iter_unpaired();
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        let map = construct_default_map();
        let iter = map.iter_unpaired();
        println!("{iter:?}");
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.clone().len(), 0);
    }

    #[test]
    fn group_map_pairing_tests() {
        let mut map: GroupMap<String, TestingStruct> = (0..10)
            .map(|i| (None, TestingStruct::from_value(i)))
            .collect();
        for i in 0..5 {
            assert_eq!(
                map.insert_left(i.to_string(), &TestingStruct::from_value(i)),
                None
            );
        }
        for i in 0..5 {
            assert!(map.pair(&i.to_string(), &TestingStruct::from_value(i + 5)));
            assert!(map.are_paired(&i.to_string(), &TestingStruct::from_value(i + 5)));
            assert!(!map.are_paired(&i.to_string(), &TestingStruct::from_value(i)));
        }
        for i in 0..10 {
            if i < 5 {
                assert!(map.are_paired(&i.to_string(), &TestingStruct::from_value(i + 5)));
                assert!(!map.are_paired(&i.to_string(), &TestingStruct::from_value(i)));
            } else {
                assert!(!map.are_paired(&i.to_string(), &TestingStruct::from_value(i)));
                assert!(map.are_paired(&(i - 5).to_string(), &TestingStruct::from_value(i)));
            }
        }
    }

    #[test]
    fn group_map_misc_tests() {
        let map = construct_default_map();
        assert!(!map.are_paired(&"0".to_string(), &TestingStruct::from_value(1)));
    }

    #[test]
    fn group_map_eq_test() {
        let map = construct_default_map();
        assert_eq!(map.clone(), construct_default_map());
        assert_eq!(construct_default_map(), construct_default_map());
    }

    #[test]
    fn group_map_fmt_tests() {
        let map = construct_default_map();
        println!("{map:?}");
    }
}
