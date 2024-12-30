#[cfg(test)]
mod tests {
    use std::{collections::HashSet, hash::Hash};

    use cycle_map::{CycleMap, InsertOptional, OptionalPair};
    use OptionalPair::*;

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

    fn construct_default_map() -> CycleMap<String, TestingStruct> {
        (0..10)
            .map(|i| (i.to_string(), TestingStruct::from_value(i)))
            .collect()
    }

    #[test]
    fn cycle_map_construction_test() {
        let map: CycleMap<String, TestingStruct> = CycleMap::new();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 0);
        let mut map = construct_default_map();
        assert_eq!(map.len(), 10);
        let cap = map.capacity();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), cap);
    }

    #[test]
    fn cycle_map_insert_test() {
        let mut map: CycleMap<u64, String> = CycleMap::with_capacity(100);
        for i in 0..100 {
            let opt = map.insert(i, i.to_string());
            assert_eq!(opt, InsertOptional::Neither);
        }
        assert_eq!(map.len(), 100);
        for (val, s) in map.iter() {
            assert_eq!(val.to_string(), *s);
            assert_eq!(str::parse::<u64>(s).expect("Unreachable"), *val);
            println!("{val}, {s}");
        }
    }

    #[test]
    fn cycle_map_get_tests() {
        let map: CycleMap<String, TestingStruct> = construct_default_map();
        assert!(map.contains_left(&0.to_string()));
        assert!(map.contains_right(&TestingStruct::from_value(0)));
        let opt = map.get_left(&TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.get_left(&TestingStruct::from_value(0));
        assert_eq!(opt, Some(&"0".to_string()));
        let opt = map.get_right(&"42".to_string());
        assert!(opt.is_none());
        let opt = map.get_right(&"0".to_string());
        assert_eq!(opt, Some(&TestingStruct::from_value(0)));
    }

    #[test]
    fn cycle_map_remove_tests() {
        // Double remove
        let mut map: CycleMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove(&"42".to_string(), &TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.remove(&"0".to_string(), &TestingStruct::from_value(0));
        assert_eq!(opt, Some(("0".to_string(), TestingStruct::from_value(0))));
        // Left remove
        let mut map: CycleMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove_via_right(&TestingStruct::from_value(42));
        assert!(opt.is_none());
        let opt = map.remove_via_right(&TestingStruct::from_value(0));
        assert_eq!(opt, Some(("0".to_string(), TestingStruct::from_value(0))));
        // Right remove
        let mut map: CycleMap<String, TestingStruct> = construct_default_map();
        let opt = map.remove_via_left(&"42".to_string());
        assert!(opt.is_none());
        let opt = map.remove_via_left(&"0".to_string());
        assert_eq!(opt, Some(("0".to_string(), TestingStruct::from_value(0))));
    }

    #[test]
    fn cycle_map_swap_left_not_found_test() {
        // Not Found
        let mut map = construct_default_map();
        let opt = map.swap_left(&"101".to_string(), "102".to_string());
        assert!(opt.is_none());
        // No collision
        let mut map = construct_default_map();
        let opt = map.swap_left(&"0".to_string(), "101".to_string());
        assert_eq!(opt, SomeLeft("0".to_string()));
        let opt = map.get_right(&"101".to_string());
        assert_eq!(opt, Some(&TestingStruct::from_value(0)));
        // With collision
        let mut map = construct_default_map();
        let opt = map.swap_left(&"0".to_string(), "1".to_string());
        assert_eq!(
            opt,
            SomeBoth(
                "0".to_string(),
                ("1".to_string(), TestingStruct::from_value(1))
            )
        );
        let opt = map.get_right(&"1".to_string());
        assert_eq!(opt, Some(&TestingStruct::from_value(0)));
    }

    #[test]
    fn cycle_map_swap_left_checked_test() {
        let mut map = construct_default_map();
        let opt = map.swap_left_checked(
            &"0".to_string(),
            &TestingStruct::from_value(1),
            "2".to_string(),
        );
        assert_eq!(opt, Neither);
        let opt = map.swap_left_checked(
            &"0".to_string(),
            &TestingStruct::from_value(0),
            "101".to_string(),
        );
        assert_eq!(opt, SomeLeft("0".to_string()));
    }

    #[test]
    fn cycle_map_swap_left_or_insert_tests() {
        let mut map = construct_default_map();
        let opt = map.swap_left_or_insert(
            &"0".to_string(),
            "101".to_string(),
            TestingStruct::from_value(0),
        );
        assert_eq!(opt, SomeLeft("0".to_string()));
        // No collision
        let mut map = construct_default_map();
        let opt = map.swap_left_or_insert(
            &"101".to_string(),
            "102".to_string(),
            TestingStruct::from_value(102),
        );
        assert_eq!(opt, Neither);
    }

    #[test]
    fn cycle_map_swap_right_not_found_test() {
        // Not Found
        let mut map = construct_default_map();
        let opt = map.swap_right(
            &TestingStruct::from_value(101),
            TestingStruct::from_value(102),
        );
        assert!(opt.is_none());
        // No collision
        let mut map = construct_default_map();
        let opt = map.swap_right(
            &TestingStruct::from_value(0),
            TestingStruct::from_value(101),
        );
        assert_eq!(opt, SomeLeft(TestingStruct::from_value(0)));
        let opt = map.get_left(&TestingStruct::from_value(101));
        assert_eq!(opt, Some(&"0".to_string()));
        // With collision
        let mut map = construct_default_map();
        let opt = map.swap_right(&TestingStruct::from_value(0), TestingStruct::from_value(1));
        assert_eq!(
            opt,
            SomeBoth(
                TestingStruct::from_value(0),
                ("1".to_string(), TestingStruct::from_value(1))
            )
        );
        let opt = map.get_left(&TestingStruct::from_value(1));
        assert_eq!(opt, Some(&"0".to_string()));
    }

    #[test]
    fn cycle_map_swap_right_checked_test() {
        let mut map = construct_default_map();
        let opt = map.swap_right_checked(
            &TestingStruct::from_value(1),
            &"0".to_string(),
            TestingStruct::from_value(2),
        );
        assert_eq!(opt, Neither);
        let opt = map.swap_right_checked(
            &TestingStruct::from_value(0),
            &"0".to_string(),
            TestingStruct::from_value(101),
        );
        assert_eq!(opt, SomeLeft(TestingStruct::from_value(0)));
    }

    #[test]
    fn cycle_map_swap_right_or_insert_tests() {
        let mut map = construct_default_map();
        let opt = map.swap_right_or_insert(
            &TestingStruct::from_value(0),
            TestingStruct::from_value(101),
            "0".to_string(),
        );
        assert_eq!(opt, SomeLeft(TestingStruct::from_value(0)));
        let mut map = construct_default_map();
        let opt = map.swap_right_or_insert(
            &TestingStruct::from_value(101),
            TestingStruct::from_value(102),
            "102".to_string(),
        );
        assert_eq!(opt, Neither);
    }

    #[test]
    fn cycle_map_retain_test() {
        let mut map: CycleMap<u64, String> = CycleMap::with_capacity(100);
        for i in 0..100 {
            let opt = map.insert(i, i.to_string());
            assert_eq!(opt, InsertOptional::Neither);
        }
        assert_eq!(map.len(), 100);
        map.retain(|x, _| x % 2 == 0);
        assert_eq!(map.len(), 50);
        for (val, s) in map.iter() {
            assert_eq!(val % 2, 0);
            println!("{val}, {s}");
        }
    }

    #[test]
    fn cycle_map_iter_tests() {
        // Main iter
        let map = construct_default_map();
        let iter = map.iter();
        println!("{iter:?}");
        assert_eq!(iter.len(), 10);
        assert_eq!(iter.clone().len(), 10);
        assert_eq!(
            iter.map(|(l, r)| (l.clone(), r.clone()))
                .collect::<CycleMap<String, TestingStruct>>(),
            map
        );
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
    fn cycle_map_drain_tests() {
        let mut map: CycleMap<u64, String> = (0..100).map(|i| (i, i.to_string())).collect();
        let cap = map.capacity();
        let other_map: CycleMap<u64, String> = map.drain().collect();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), cap);
        assert_eq!(other_map.len(), 100);
        let mut map: CycleMap<u64, String> = (0..100).map(|i| (i, i.to_string())).collect();
        let other_map: CycleMap<u64, String> = map.drain_filter(|l, _| l % 2 == 0).collect();
        assert_eq!(map.len(), 50);
        assert_eq!(other_map.len(), 50);
        for (l, _) in other_map.iter() {
            assert!(l % 2 == 0);
        }
    }

    #[test]
    fn cycle_map_eq_test() {
        let map = construct_default_map();
        assert_eq!(map.clone(), construct_default_map());
        assert_eq!(construct_default_map(), construct_default_map());
    }

    #[test]
    fn cycle_map_shrink_tests() {
        let mut map: CycleMap<i32, i32> = CycleMap::with_capacity(100);
        let cap = map.capacity();
        map.insert(1, 2);
        map.insert(3, 4);
        assert_eq!(map.capacity(), cap);
        map.shrink_to(10);
        assert!(map.capacity() >= 10);
        assert!(map.capacity() <= cap);
        map.shrink_to(0);
        assert!(map.capacity() >= 2);
        assert!(map.capacity() <= cap);
        map.shrink_to(10);
        assert!(map.capacity() >= 2);
        assert!(map.capacity() <= cap);
        let mut map: CycleMap<i32, i32> = CycleMap::with_capacity(100);
        let cap = map.capacity();
        map.insert(1, 2);
        map.insert(3, 4);
        assert_eq!(map.capacity(), cap);
        assert!(map.capacity() >= 10);
        assert!(map.capacity() <= cap);
        map.shrink_to_fit();
        assert!(map.capacity() >= 2);
        assert!(map.capacity() <= 10);
    }

    #[test]
    fn cycle_map_reserve_tests() {
        let mut map: CycleMap<&str, i32> = CycleMap::new();
        let old_cap = map.capacity();
        assert_eq!(old_cap, 0);
        map.reserve(10);
        assert!(old_cap != map.capacity());

        use cycle_map::CycleMap;
        let mut map: CycleMap<&str, i32> = CycleMap::new();
        let old_cap = map.capacity();
        assert_eq!(old_cap, 0);
        let res = map.try_reserve(10);
        assert!(res.is_ok());
        assert!(old_cap != map.capacity());
    }

    #[test]
    fn cycle_map_fmt_tests() {
        let map = construct_default_map();
        println!("{map:?}");
    }

    #[test]
    fn cycle_map_misc_tests() {
        let map = construct_default_map();
        let _hasher = map.hasher();
        assert!(!map.are_paired(&"0".to_string(), &TestingStruct::from_value(1)));
    }

    #[test]
    fn cycle_map_hash_collision() {
        #[derive(Debug, PartialEq, Eq)]
        struct Collide(usize);

        impl Hash for Collide {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                (42).hash(state);
            }
        }

        let map: CycleMap<_, _> = (0..100).map(|i| (Collide(i), i.to_string())).collect();
        assert_eq!(map.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get_right(&Collide(i)).unwrap(), &i.to_string());
        }
    }
}
