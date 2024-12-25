use std::{
    borrow::Borrow,
    default::Default,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    marker::PhantomData,
    vec,
};

use hashbrown::{hash_table, DefaultHashBuilder, HashTable, TryReserveError};

use crate::{optionals::*, utils::*};
use OptionalPair::*;

#[cfg(doc)]
use crate::CycleMap;

// Contains a value and the hash of the item that the value maps to.
pub(crate) struct MappingPair<T> {
    pub(crate) value: T,
    pub(crate) hash: Option<u64>,
    pub(crate) id: u64,
}

pub(crate) fn equivalent_key<K, Q: PartialEq<K>>(k: &Q) -> impl '_ + Fn(&MappingPair<K>) -> bool {
    move |x| k.eq(&x.value)
}

pub(crate) fn hash_and_id<Q: PartialEq>(hash: u64, id: u64) -> impl Fn(&MappingPair<Q>) -> bool {
    move |x| id == x.id && Some(hash) == x.hash
}

// To be safe, use `hash_and_id` whenever possible
pub(crate) fn just_id<Q: PartialEq>(id: u64) -> impl Fn(&MappingPair<Q>) -> bool {
    move |x| id == x.id
}

/// A map similar to [`CycleMap`] but items in either set can unpaired.
///
/// `PartialCycleMap` takes loosens the pairing requirement of a `CycleMap`, but the other
/// requirements (namely that values must implement [`Eq`] and [`Hash`]) remain.
///
/// The enum [`OptionalPair`] is used extensively throughout the `PartialCycleMap` since full pairs
/// often can't be be guaranteed. `OptionalPair` helps to express this by giving a more ergonomic
/// feel to its equivalent representation, `(Option<A>, Option<B>)` or `Option<(Option<A>,
/// Option<B>)`.
///
/// Note: While a `PartialCycleMap` can do everything that a `CycleMap` can, it is generally less
/// efficient. Many more checks need to be done since every item can't be assumed to be paired.
/// When possible, it is generally better to use a `CycleMap`.
///
/// # Examples
/// ```
/// use cycle_map::{PartialCycleMap, OptionalPair};
/// use OptionalPair::*;
///
/// let values: Vec<OptionalPair<&str, u64>> =
///              vec![ SomeBoth("zero", 0), SomeBoth("one", 1), SomeBoth("two", 2),
///                    SomeBoth("three", 3), SomeBoth("four", 4), SomeBoth("five", 5),
///                    SomeLeft("six"), SomeLeft("seven"), SomeLeft("eight"),
///                    SomeLeft("nine"), ];
///
/// let mut converter: PartialCycleMap<&str, u64> = values.iter().cloned().collect();
///
/// // The map should contain 10 items in the left set ...
/// assert_eq!(converter.len_left(), 10);
/// // ... and 6 in the right set
/// assert_eq!(converter.len_right(), 6);
///
/// // See if your value number is here
/// if converter.contains_right(&42) {
///     println!( "I know the answer to life!!" );
/// }
///
/// // Get a value from either side (if paired)!
/// assert!(!converter.contains_right(&7));
/// assert_eq!(converter.get_left(&7), None);
/// assert_eq!(converter.get_right(&"three"), Some(&3));
///
/// // Items can be unpaired by removal or by just unpairing them!
/// assert!(converter.are_paired(&"four", &4));
/// assert_eq!(converter.remove_right(&4), Some(4));
/// assert!(converter.unpair(&"three", &3));
/// assert!(!converter.are_paired(&"three", &3));
///
/// // Bring items together!
/// assert!(converter.pair(&"three", &3));
/// assert!(converter.are_paired(&"three", &3));
/// ```
pub struct PartialCycleMap<L, R, St = DefaultHashBuilder> {
    pub(crate) hash_builder: St,
    pub(crate) counter: u64,
    left_set: HashTable<MappingPair<L>>,
    right_set: HashTable<MappingPair<R>>,
}

impl<L, R> PartialCycleMap<L, R, DefaultHashBuilder> {
    #[inline]
    /// Creates a new `PartialCycleMap`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let map: PartialCycleMap<u64, String> = PartialCycleMap::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    /// Creates a new, empty `PartialCycleMap` with inner sets that each have at least the given capacity
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let map: PartialCycleMap<u64, String> = PartialCycleMap::with_capacity(100);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<L, R, S> PartialCycleMap<L, R, S>
where
    L: Eq + Hash,
    R: Eq + Hash,
    S: BuildHasher,
{
    /// Adds a pair of items to the map.
    ///
    /// Should the left element be equal to another left element, the (optional) pair containing
    /// the old left item is removed and returned. The same goes for the new right element.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// // Neither 5 nor "5" is in map
    /// let op = map.insert(5, 5.to_string());
    /// assert_eq!(op, (Neither, Neither));
    ///
    /// // 0 is in the map, its old pairing is removed
    /// let op = map.insert(0, 6.to_string());
    /// assert_eq!(op, (SomeBoth(0, 0.to_string()), Neither));
    ///
    /// // "1" is in the map, its old pairing is removed
    /// let op = map.insert(7, 1.to_string());
    /// assert_eq!(op, (Neither, SomeBoth(1, 1.to_string())));
    ///
    /// // Both 2 and "3" are in the map, so their old pairings are removed
    /// let op = map.insert(2, 3.to_string());
    /// assert_eq!(op, (SomeBoth(2, 2.to_string()), SomeBoth(3, 3.to_string())));
    /// ```
    pub fn insert(&mut self, left: L, right: R) -> (OptionalPair<L, R>, OptionalPair<L, R>) {
        let opt_from_left = self.remove_via_left(&left);
        let opt_from_right = self.remove_via_right(&right);
        let digest = (opt_from_left, opt_from_right);
        let l_hash = make_hash::<L, S>(&self.hash_builder, &left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, &right);
        let left_pairing = MappingPair {
            value: left,
            hash: Some(r_hash),
            id: self.counter,
        };
        let right_pairing = MappingPair {
            value: right,
            hash: Some(l_hash),
            id: self.counter,
        };
        self.counter += 1;
        self.left_set.insert_unique(
            l_hash,
            left_pairing,
            make_hasher::<MappingPair<L>, S>(&self.hash_builder),
        );
        self.right_set.insert_unique(
            r_hash,
            right_pairing,
            make_hasher::<MappingPair<R>, S>(&self.hash_builder),
        );
        digest
    }

    /// Adds an item to the left set of the map.
    ///
    /// Should this item be equal to another, the old item is removed. If that item was paired, the
    /// pair is removed. In either case, the (optional) pair is returned.
    /// a right item, the pair is removed.
    ///
    /// Note: If you want to swap the left item in a pair, use the [`swap_left`] method.
    ///
    /// Also Note: This method will never return the `SomeLeft` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// // 5 is not in map
    /// let op = map.insert_left(5);
    /// assert_eq!(op, Neither);
    ///
    /// // 5 is in the map, but is unpaired
    /// let op = map.insert_left(5);
    /// assert_eq!(op, SomeLeft(5));
    ///
    /// // 0 is in the map, its old pairing is removed
    /// let op = map.insert_left(0);
    /// assert_eq!(op, SomeBoth(0, 0.to_string()));
    /// ```
    ///
    /// [`swap_left`]: struct.PartialCycleMap.html#method.swap_left
    pub fn insert_left(&mut self, left: L) -> OptionalPair<L, R> {
        let opt_from_left = self.remove_via_left(&left);
        let digest = opt_from_left;
        let l_hash = make_hash::<L, S>(&self.hash_builder, &left);
        let left_pairing = MappingPair {
            value: left,
            hash: None,
            id: self.counter,
        };
        self.counter += 1;
        self.left_set.insert_unique(
            l_hash,
            left_pairing,
            make_hasher::<MappingPair<L>, S>(&self.hash_builder),
        );
        digest
    }

    /// Adds an item to the right set of the map.
    ///
    /// Should this item be equal to another, the old item is removed. If that item was paired with
    /// a left item, the pair is removed.
    ///
    /// Note: If you want to swap the right item in a pair, use the [`swap_right`] method.
    ///
    /// Also Note: This method will never return the `SomeLeft` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// // "5" is not in map
    /// let op = map.insert_right("5".to_string());
    /// assert_eq!(op, Neither);
    ///
    /// // "5" is in the map, but is unpaired
    /// let op = map.insert_right("5".to_string());
    /// assert_eq!(op, SomeRight("5".to_string()));
    ///
    /// // "0" is in the map, its old pairing is removed
    /// let op = map.insert_right("0".to_string());
    /// assert_eq!(op, SomeBoth(0, "0".to_string()));
    /// ```
    ///
    /// [`swap_right`]: struct.PartialCycleMap.html#method.swap_right
    pub fn insert_right(&mut self, right: R) -> OptionalPair<L, R> {
        let opt_from_right = self.remove_via_right(&right);
        let digest = opt_from_right;
        let r_hash = make_hash::<R, S>(&self.hash_builder, &right);
        let right_pairing = MappingPair {
            value: right,
            hash: None,
            id: self.counter,
        };
        self.counter += 1;
        self.right_set.insert_unique(
            r_hash,
            right_pairing,
            make_hasher::<MappingPair<R>, S>(&self.hash_builder),
        );
        digest
    }

    /// Pairs two existing items in the map. Returns `true` if they were successfully paired.
    /// Returns `false` if either item can not be found or if either items is already paired.
    ///
    /// Use [`pair_forced`] if you want to break the existing pairings.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = PartialCycleMap::new();
    /// map.insert_left(1);
    /// map.insert_left(2);
    /// map.insert_right(1.to_string());
    /// map.insert_right(2.to_string());
    ///
    /// // Pair 1 and "1"
    /// assert!(!map.are_paired(&1, &1.to_string()));
    /// assert!(map.pair(&1, &1.to_string()));
    /// assert!(map.are_paired(&1, &1.to_string()));
    ///
    /// // Note, we can't simply pair 1 and "2" since 1 is paired
    /// assert!(!map.pair(&1, &2.to_string()));
    /// ```
    ///
    /// [`pair_forced`]: struct.PartialCycleMap.html#method.pair_forced
    pub fn pair(&mut self, left: &L, right: &R) -> bool {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let opt_left = self.left_set.find_mut(l_hash, equivalent_key(left));
        let opt_right = self.right_set.find_mut(r_hash, equivalent_key(right));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => match (left.hash, right.hash) {
                (None, None) => {
                    left.hash = Some(r_hash);
                    right.hash = Some(l_hash);
                    right.id = left.id;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Pairs two existing items in the map. Items that are paired become unpaired but remain in
    /// the map. References to items that become unpaired are returned.
    ///
    /// Use [`pair_forced_remove`] if you want to remove the items that become unpaired.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = PartialCycleMap::new();
    /// map.insert_left(1);
    /// map.insert_left(2);
    /// map.insert_right(1.to_string());
    /// map.insert_right(2.to_string());
    ///
    /// // Pair 1 and "1"
    /// assert!(!map.are_paired(&1, &1.to_string()));
    /// let op = map.pair_forced(&1, &1.to_string());
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&1, &1.to_string()));
    ///
    /// // Note, we can't simply pair 1 and "2" since 1 is paired
    /// let op = map.pair_forced(&1, &2.to_string());
    /// assert_eq!(op, SomeRight(&"1".to_string()));
    /// assert!(map.are_paired(&1, &2.to_string()));
    /// assert!(!map.are_paired(&1, &1.to_string()));
    /// ```
    ///
    /// [`pair_forced_remove`]: struct.PartialCycleMap.html#method.pair_forced_remove
    pub fn pair_forced(&mut self, l: &L, r: &R) -> OptionalPair<&L, &R> {
        if self.are_paired(l, r) {
            return Neither;
        }
        let l_hash = make_hash::<L, S>(&self.hash_builder, l);
        let r_hash = make_hash::<R, S>(&self.hash_builder, r);
        let opt_left = self.left_set.find_mut(l_hash, equivalent_key(l));
        let opt_right = self.right_set.find_mut(r_hash, equivalent_key(r));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => match (left.hash, right.hash) {
                (None, None) => {
                    left.hash = Some(r_hash);
                    right.hash = Some(l_hash);
                    right.id = left.id;
                    Neither
                }
                (Some(lp_hash), None) => {
                    left.hash = Some(r_hash);
                    right.hash = Some(l_hash);
                    let old_id = left.id;
                    // Here, we give the left item the new id to avoid a collision in the right set
                    left.id = right.id;
                    self.right_set
                        .find_mut(lp_hash, hash_and_id(l_hash, old_id))
                        .unwrap()
                        .hash = None;
                    SomeRight(&self.right_set.find(lp_hash, just_id(old_id)).unwrap().value)
                }
                (None, Some(rp_hash)) => {
                    left.hash = Some(r_hash);
                    right.hash = Some(l_hash);
                    let old_id = right.id;
                    // Here, we give the right item the new id to avoid a collision in the left set
                    right.id = left.id;
                    self.left_set
                        .find_mut(rp_hash, hash_and_id(r_hash, old_id))
                        .unwrap()
                        .hash = None;
                    SomeLeft(&self.left_set.find(rp_hash, just_id(old_id)).unwrap().value)
                }
                (Some(lp_hash), Some(rp_hash)) => {
                    left.hash = Some(r_hash);
                    right.hash = Some(l_hash);
                    let old_l_id = left.id;
                    let old_r_id = right.id;
                    // Here, we give the pair a new id to avoid collisions in both sets
                    left.id = self.counter;
                    right.id = self.counter;
                    self.counter += 1;
                    self.left_set
                        .find_mut(rp_hash, hash_and_id(r_hash, old_r_id))
                        .unwrap()
                        .hash = None;
                    self.right_set
                        .find_mut(lp_hash, hash_and_id(l_hash, old_l_id))
                        .unwrap()
                        .hash = None;
                    SomeBoth(
                        &self
                            .left_set
                            .find(rp_hash, just_id(old_r_id))
                            .unwrap()
                            .value,
                        &self
                            .right_set
                            .find(lp_hash, just_id(old_l_id))
                            .unwrap()
                            .value,
                    )
                }
            },
            _ => Neither,
        }
    }

    /// Pairs two existing items in the map. Items that are paired become unpaired and are removed
    /// from the map. The old items are returned.
    ///
    /// This is equivalent to chained calls to [`swap_left`] and [`swap_right`].
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = PartialCycleMap::new();
    /// map.insert_left(1);
    /// map.insert_left(2);
    /// map.insert_right(1.to_string());
    /// map.insert_right(2.to_string());
    ///
    /// // Pair 1 and "1"
    /// assert!(!map.are_paired(&1, &1.to_string()));
    /// let op = map.pair_forced_remove(&1, &1.to_string());
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&1, &1.to_string()));
    ///
    /// // Note, we can't simply pair 1 and "2" since 1 is paired
    /// let op = map.pair_forced_remove(&1, &2.to_string());
    /// assert_eq!(op, SomeRight("1".to_string()));
    /// assert!(map.are_paired(&1, &2.to_string()));
    /// assert!(!map.are_paired(&1, &1.to_string()));
    /// ```
    ///
    /// [`swap_left`]: struct.PartialCycleMap.html#method.swap_left
    /// [`swap_right`]: struct.PartialCycleMap.html#method.swap_right
    pub fn pair_forced_remove(&mut self, l: &L, r: &R) -> OptionalPair<L, R> {
        if self.are_paired(l, r) {
            return Neither;
        }
        let l_hash = make_hash(&self.hash_builder, l);
        let r_hash = make_hash(&self.hash_builder, r);
        let opt_left = self.left_set.find_mut(l_hash, equivalent_key(l));
        let opt_right = self.right_set.find_mut(r_hash, equivalent_key(r));
        let (Some(left), Some(right)) = (opt_left, opt_right) else {
            // TODO: This should probably be an error rather than Neither because
            return Neither;
        };
        match (left.hash, right.hash) {
            (None, None) => {
                left.hash = Some(r_hash);
                right.hash = Some(l_hash);
                right.id = left.id;
                Neither
            }
            (Some(lp_hash), None) => {
                left.hash = Some(r_hash);
                right.hash = Some(l_hash);
                let old_id = left.id;
                // Here, we give the left item the new id to avoid a collision in the right set
                left.id = right.id;
                let Ok(entry) = self
                    .right_set
                    .find_entry(lp_hash, hash_and_id(l_hash, old_id))
                else {
                    unreachable!("TODO");
                };
                SomeRight(entry.remove().0.value)
            }
            (None, Some(rp_hash)) => {
                left.hash = Some(r_hash);
                right.hash = Some(l_hash);
                let old_id = right.id;
                // Here, we give the left item the new id to avoid a collision in the right set
                right.id = left.id;
                let Ok(entry) = self
                    .left_set
                    .find_entry(rp_hash, hash_and_id(r_hash, old_id))
                else {
                    unreachable!("TODO")
                };
                SomeLeft(entry.remove().0.value)
            }
            (Some(lp_hash), Some(rp_hash)) => {
                left.hash = Some(r_hash);
                right.hash = Some(l_hash);
                let old_l_id = left.id;
                let old_r_id = right.id;
                // Here, we give the pair a new id to avoid collisions in both sets
                left.id = self.counter;
                right.id = self.counter;
                self.counter += 1;
                let Ok(left_entry) = self
                    .left_set
                    .find_entry(rp_hash, hash_and_id(r_hash, old_r_id))
                else {
                    unreachable!("TODO")
                };
                let Ok(right_entry) = self
                    .right_set
                    .find_entry(lp_hash, hash_and_id(l_hash, old_l_id))
                else {
                    unreachable!("TODO");
                };
                SomeBoth(left_entry.remove().0.value, right_entry.remove().0.value)
            }
        }
    }

    /// Unpairs two existing items in the map. Returns `true` if they were successfully unpaired.
    /// Returns `false` if either item can not be found or if they aren't paired.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, &str> = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// map.insert_right("2");
    ///
    /// // Unpair (1, "1")
    /// assert!(map.unpair(&1, &"1"));
    /// assert!(!map.are_paired(&1, &"1"));
    ///
    /// // Note, we can't unpair things that are already unpaired
    /// assert!(!map.are_paired(&2, &"2"));
    /// assert!(!map.unpair(&2, &"2"));
    /// ```
    pub fn unpair(&mut self, left: &L, right: &R) -> bool {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let opt_left = self.left_set.find_mut(l_hash, equivalent_key(left));
        let opt_right = self.right_set.find_mut(r_hash, equivalent_key(right));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => match (left.hash, right.hash) {
                (Some(l_h), Some(r_h)) => {
                    if l_hash == r_h && r_hash == l_h {
                        left.hash = None;
                        right.hash = None;
                        right.id = self.counter;
                        self.counter += 1;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Determines if an item in the left set is paired.
    ///
    /// Returns false if the item isn't found or is unpaired. Returns true otherwise.
    pub fn is_left_paired(&self, left: &L) -> bool {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let opt_left = self.left_set.find(l_hash, equivalent_key(left));
        match opt_left {
            Some(l) => l.hash.is_some(),
            None => false,
        }
    }

    /// Determines if an item in the right set is paired.
    ///
    /// Returns false if the item isn't found or is unpaired. Returns true otherwise.
    pub fn is_right_paired(&self, right: &R) -> bool {
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let opt_right = self.right_set.find(r_hash, equivalent_key(right));
        match opt_right {
            Some(r) => r.hash.is_some(),
            None => false,
        }
    }

    /// Returns `true` if both items are in the map and are paired together; otherwise, returns
    /// `false`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// assert!(map.are_paired(&1, &"1"));
    /// assert!(!map.are_paired(&2, &"1"));
    /// assert!(!map.are_paired(&2, &"2"));
    /// ```
    pub fn are_paired(&self, left: &L, right: &R) -> bool {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let opt_left = self.left_set.find(l_hash, equivalent_key(left));
        let opt_right = self.right_set.find(r_hash, equivalent_key(right));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => {
                left.id == right.id && Some(l_hash) == right.hash && Some(r_hash) == left.hash
            }
            _ => false,
        }
    }

    /// Returns `true` if the item is found and `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// assert!(map.contains_left(&1));
    /// assert!(map.contains_left(&2));
    /// assert!(!map.contains_left(&3));
    /// ```
    pub fn contains_left(&self, left: &L) -> bool {
        let hash = make_hash::<L, S>(&self.hash_builder, left);
        self.left_set.find(hash, equivalent_key(left)).is_some()
    }

    /// Returns `true` if the item is found and `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_right("2");
    /// assert!(map.contains_right(&"1"));
    /// assert!(map.contains_right(&"2"));
    /// assert!(!map.contains_right(&"3"));
    /// ```
    pub fn contains_right(&self, right: &R) -> bool {
        let hash = make_hash::<R, S>(&self.hash_builder, right);
        self.right_set.find(hash, equivalent_key(right)).is_some()
    }

    /// Removes and returns the give pair from the map provided that they are paired together.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.remove(&1, &"1"), Some((1, "1")));
    /// assert_eq!(map.remove(&1, &"1"), None);
    /// ```
    pub fn remove(&mut self, left: &L, right: &R) -> Option<(L, R)> {
        self.are_paired(left, right).then(|| {
            let SomeBoth(left, right) = self.remove_via_left(left) else {
                unreachable!("TODO");
            };
            (left, right)
        })
    }

    /// Removes and returns the given item from the left set and unpairs its associated item if it
    /// is paired.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// assert_eq!(map.remove_left(&1), Some(1));
    /// assert!(map.contains_right(&"1"));
    /// ```
    pub fn remove_left(&mut self, item: &L) -> Option<L> {
        let l_hash = make_hash(&self.hash_builder, item);
        let item = self
            .left_set
            .find_entry(l_hash, equivalent_key(item))
            .ok()?
            .remove()
            .0;
        if let Some(hash) = item.hash {
            self.right_set
                .find_mut(hash, just_id(item.id))
                .unwrap()
                .hash = None;
        }
        Some(item.value)
    }

    /// Removes and returns the given item from the left set and, if it exists, its associated item
    /// from the right set.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// assert_eq!(map.remove_via_left(&1), SomeBoth(1, "1"));
    /// assert_eq!(map.remove_via_left(&2), SomeLeft(2));
    /// assert_eq!(map.remove_via_left(&3), Neither);
    /// ```
    pub fn remove_via_left(&mut self, item: &L) -> OptionalPair<L, R> {
        let l_hash = make_hash::<L, S>(&self.hash_builder, item);
        let Ok(entry) = self.left_set.find_entry(l_hash, equivalent_key(item)) else {
            return Neither;
        };
        let left_pairing = entry.remove().0;
        let right_value = left_pairing.hash.map(|hash| {
            let Ok(entry) = self
                .right_set
                .find_entry(hash, hash_and_id(l_hash, left_pairing.id))
            else {
                unreachable!("TODO")
            };
            entry.remove().0.value
        });
        OptionalPair::from((Some(left_pairing.value), right_value))
    }

    /// Removes and returns the given item from the left set and unpairs its associated item if it
    /// is paired.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_right("2");
    /// assert_eq!(map.remove_right(&"1"), Some("1"));
    /// assert!(map.contains_left(&1));
    /// ```
    pub fn remove_right(&mut self, item: &R) -> Option<R> {
        let r_hash = make_hash(&self.hash_builder, item);
        let item = self
            .right_set
            .find_entry(r_hash, equivalent_key(item))
            .ok()?
            .remove()
            .0;
        if let Some(hash) = item.hash {
            self.left_set.find_mut(hash, just_id(item.id)).unwrap().hash = None;
        }
        Some(item.value)
    }

    /// Removes the given item from the right set and its associated item from the left set
    ///
    /// Note: This method will never return the `SomeLeft` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_right("2");
    /// assert_eq!(map.remove_via_right(&"1"), SomeBoth(1, "1"));
    /// assert_eq!(map.remove_via_right(&"2"), SomeRight("2"));
    /// assert_eq!(map.remove_via_right(&"3"), Neither);
    /// ```
    pub fn remove_via_right(&mut self, item: &R) -> OptionalPair<L, R> {
        let r_hash = make_hash::<R, S>(&self.hash_builder, item);
        let Ok(entry) = self.right_set.find_entry(r_hash, equivalent_key(item)) else {
            return Neither;
        };
        let right_pairing = entry.remove().0;
        let left_value = right_pairing.hash.map(|hash| {
            let Ok(entry) = self
                .left_set
                .find_entry(hash, hash_and_id(r_hash, right_pairing.id))
            else {
                unreachable!("TODO")
            };
            entry.remove().0.value
        });
        OptionalPair::from((left_value, Some(right_pairing.extract())))
    }

    /// Swaps an item in the left set with another item, remaps the old item's associated right
    /// item, and returns the old left item.
    ///
    /// If there is another item in the left set that is equal to the new left item which is paired
    /// to another right item, that cycle is removed and returned
    ///
    /// `Neither` is returned if the old item isn't in the map.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert(2, "2");
    /// map.insert_left(3);
    /// // Swap 1 in the pair (1, "1") with 0
    /// assert_eq!(map.swap_left(&1, 0), SomeLeft(1));
    ///
    /// // Swap 2 in the pair (2, "2") with 3, which removes the unpaired 3.
    /// assert_eq!(map.swap_left(&2, 3), SomeBoth(2, SomeLeft(3)));
    ///
    /// // 4 is not in the map, so there is nothing to swap.
    /// assert_eq!(map.swap_left(&4, 5), Neither);
    /// ```
    pub fn swap_left(&mut self, old: &L, new: L) -> OptionalPair<L, OptionalPair<L, R>> {
        // Check for Eq left item and remove that cycle if it exists
        let new_l_hash = make_hash(&self.hash_builder, &new);
        let eq_opt = self.swap_left_eq_check(old, &new, new_l_hash);
        // Find the old left pairing
        let old_l_hash = make_hash(&self.hash_builder, old);
        let Some(l_pairing) = self.left_set.find(old_l_hash, equivalent_key(old)) else {
            return Neither;
        };
        if let Some(hash) = l_pairing.hash {
            // Use old left pairing to find right pairing
            let r_pairing: &mut MappingPair<R> = self
                .right_set
                .find_mut(hash, hash_and_id(old_l_hash, l_pairing.id))
                .unwrap();
            // Updated right pairing
            r_pairing.hash = Some(new_l_hash);
        }
        // Create new left pairing
        let new_left_pairing = MappingPair {
            value: new,
            hash: l_pairing.hash,
            id: l_pairing.id,
        };
        // Remove old left pairing
        let Ok(entry) = self.left_set.find_entry(old_l_hash, equivalent_key(old)) else {
            unreachable!("TODO")
        };
        let old_left_item = entry.remove().0.value;
        // Insert new left pairing
        self.left_set.insert_unique(
            new_l_hash,
            new_left_pairing,
            make_hasher::<MappingPair<L>, S>(&self.hash_builder),
        );
        // Return old left pairing
        if eq_opt.is_none() {
            SomeLeft(old_left_item)
        } else {
            SomeBoth(old_left_item, eq_opt)
        }
    }

    /// Does what [`swap_left`] does, but fails to swap and returns `Neither` if the old item isn't
    /// paired to the given right item.
    ///
    /// # Examples
    /// The following are equivalent
    /// ```rust
    /// # use cycle_map::{PartialCycleMap, OptionalPair::*};
    /// # let mut map = PartialCycleMap::new();
    /// let op_one = map.swap_left_checked(&1, &"1", 2);
    ///
    /// let op_two = if map.are_paired(&1, &"1") {
    ///     map.swap_left(&1, 2)
    /// } else {
    ///     Neither
    /// };
    ///
    /// assert_eq!(op_one, op_two);
    /// ```
    ///
    /// [`swap_left`]: struct.PartialCycleMap.html#method.swap_left
    pub fn swap_left_checked(
        &mut self,
        old: &L,
        expected: &R,
        new: L,
    ) -> OptionalPair<L, OptionalPair<L, R>> {
        // Check if old and expected are paired
        if !self.are_paired(old, expected) {
            return Neither;
        }
        self.swap_left(old, new)
    }

    /// Does what [`swap_left`] does, but inserts a new pair if the old left item isn't in the map.
    /// None is returned on insert.
    ///
    /// # Examples
    /// The following are equivalent
    /// ```rust
    /// # use cycle_map::{PartialCycleMap, OptionalPair::*};
    /// # let mut map = PartialCycleMap::new();
    /// map.swap_left_or_insert(&1, 2, "1");
    ///
    /// if !map.contains_left(&1) {
    ///     map.swap_left(&1, 2);
    /// } else {
    ///     map.insert(2, "1");
    /// }
    /// ```
    ///
    /// [`swap_left`]: struct.PartialCycleMap.html#method.swap_left
    pub fn swap_left_or_insert(
        &mut self,
        old: &L,
        new: L,
        to_insert: R,
    ) -> OptionalPair<L, OptionalPair<L, R>> {
        let old_l_hash = make_hash::<L, S>(&self.hash_builder, old);
        if self
            .left_set
            .find(old_l_hash, equivalent_key(old))
            .is_some()
        {
            self.swap_left(old, new)
        } else {
            // TODO: Do further verification on this. All cases _should_ be covered here
            match self.insert(new, to_insert) {
                (Neither, Neither) => Neither,
                (Neither, pair) => SomeRight(pair),
                _ => {
                    unreachable!("There isn't a left item")
                }
            }
        }
    }

    /// Pair of the collision checks done in the swap left methods
    fn swap_left_eq_check(&mut self, old: &L, new: &L, new_hash: u64) -> OptionalPair<L, R> {
        let opt = self.left_set.find(new_hash, equivalent_key(new));
        if opt.is_some() && new != old {
            // Remove the problem cycle
            self.remove_via_left(new)
        } else {
            // If old and new are the same, they we are updating an cycle
            Neither
        }
    }

    /// Swaps an item in the right set with another item, remaps the old item's associated left
    /// item, and returns the old right item
    ///
    /// `Neither` is returned if the old item isn't in the map.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert(2, "2");
    /// map.insert_right("3");
    /// // Swap 1 in the pair (1, "1") with 0
    /// assert_eq!(map.swap_right(&"1", "0"), SomeLeft("1"));
    ///
    /// // Swap 2 in the pair (2, "2") with 3, which removes the unpaired 3.
    /// assert_eq!(map.swap_right(&"2", "3"), SomeBoth("2", SomeRight("3")));
    ///
    /// // 4 is not in the map, so there is nothing to swap.
    /// assert_eq!(map.swap_right(&"4", "5"), Neither);
    /// ```
    pub fn swap_right(&mut self, old: &R, new: R) -> OptionalPair<R, OptionalPair<L, R>> {
        // Check for Eq left item and remove that cycle if it exists
        let new_r_hash = make_hash(&self.hash_builder, &new);
        let eq_opt = self.swap_right_eq_check(old, &new, new_r_hash);
        // Find the old right pairing
        let old_r_hash = make_hash(&self.hash_builder, old);
        let Some(r_pairing) = self.right_set.find(old_r_hash, equivalent_key(old)) else {
            return Neither;
        };
        if let Some(hash) = r_pairing.hash {
            // Use old right pairing to find the left pairing
            let l_pairing = self
                .left_set
                .find_mut(hash, hash_and_id(old_r_hash, r_pairing.id))
                .unwrap();
            // Updated left pairing
            l_pairing.hash = Some(new_r_hash);
        }
        // Create new right pairing
        let new_right_pairing = MappingPair {
            value: new,
            hash: r_pairing.hash,
            id: r_pairing.id,
        };
        // Remove old right pairing
        let Ok(entry) = self.right_set.find_entry(old_r_hash, equivalent_key(old)) else {
            unreachable!("TODO");
        };
        let old_right_item = entry.remove().0.value;
        // Insert new right pairing
        self.right_set.insert_unique(
            new_r_hash,
            new_right_pairing,
            make_hasher(&self.hash_builder),
        );
        // Return old right pairing
        if eq_opt.is_none() {
            SomeLeft(old_right_item)
        } else {
            SomeBoth(old_right_item, eq_opt)
        }
    }

    /// Does what [`swap_right`] does, but fails to swap if the old item isn't paired to the given
    /// left item.
    ///
    /// # Examples
    /// The following are equivalent
    /// ```rust
    /// # use cycle_map::{PartialCycleMap, OptionalPair::*};
    /// # let mut map = PartialCycleMap::new();
    /// let op_one = map.swap_right_checked(&"1", &1, "2");
    ///
    /// let op_two = if map.are_paired(&1, &"1") {
    ///     map.swap_right(&"1", "2")
    /// } else {
    ///     Neither
    /// };
    ///
    /// assert_eq!(op_one, op_two);
    /// ```
    ///
    /// [`swap_right`]: struct.PartialCycleMap.html#method.swap_right
    pub fn swap_right_checked(
        &mut self,
        old: &R,
        expected: &L,
        new: R,
    ) -> OptionalPair<R, OptionalPair<L, R>> {
        // Check if old and expected are paired
        if !self.are_paired(expected, old) {
            return Neither;
        } // Things can be removed after this point
        self.swap_right(old, new)
    }

    /// Does what [`swap_right`] does, but inserts a new pair if the old right item isn't in the map
    /// None is returned on insert.
    ///
    /// # Examples
    /// The following are equivalent
    /// ```rust
    /// # use cycle_map::{PartialCycleMap, OptionalPair::*};
    /// # let mut map = PartialCycleMap::new();
    /// map.swap_right_or_insert(&"1", "2", 1);
    ///
    /// if !map.contains_right(&"1") {
    ///     map.swap_right(&"1", "2");
    /// } else {
    ///     map.insert(2, "1");
    /// }
    /// ```
    ///
    /// [`swap_right`]: struct.PartialCycleMap.html#method.swap_right
    pub fn swap_right_or_insert(
        &mut self,
        old: &R,
        new: R,
        to_insert: L,
    ) -> OptionalPair<R, OptionalPair<L, R>> {
        // Find the old right pairing
        let old_r_hash = make_hash::<R, S>(&self.hash_builder, old);
        if self
            .right_set
            .find(old_r_hash, equivalent_key(old))
            .is_some()
        {
            self.swap_right(old, new)
        } else {
            // TODO: Do further verification on this. All cases _should_ be covered here
            match self.insert(to_insert, new) {
                (Neither, Neither) => Neither,
                (pair, Neither) => SomeRight(pair),
                _ => {
                    unreachable!("There isn't a left item")
                }
            }
        }
    }

    /// Pair of the collision checks done in the swap left methods
    fn swap_right_eq_check(&mut self, old: &R, new: &R, new_hash: u64) -> OptionalPair<L, R> {
        let opt = self.right_set.find(new_hash, equivalent_key(new));
        if opt.is_some() && new != old {
            // Remove the problem cycle
            self.remove_via_right(new)
        } else {
            // If old and new are the same, they we are updating an cycle
            Neither
        }
    }

    /// Gets a reference to an item in the left set using an item in the right set.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_right("2");
    /// assert_eq!(map.get_left(&"1"), Some(&1));
    /// assert_eq!(map.get_left(&"2"), None);
    /// assert_eq!(map.get_left(&"3"), None);
    /// ```
    pub fn get_left<Q>(&self, item: &Q) -> Option<&L>
    where
        R: Borrow<Q>,
        Q: Hash + Eq + PartialEq<R>,
    {
        let r_hash = make_hash::<_, S>(&self.hash_builder, item);
        let right_pairing: &MappingPair<R> = self.get_right_inner_with_hash(item, r_hash)?;
        let hash = right_pairing.hash?;
        match self
            .left_set
            .find(hash, hash_and_id(r_hash, right_pairing.id))
        {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    /// Gets a reference to an item in the right set using an item in the left set.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// assert_eq!(map.get_right(&1), Some(&"1"));
    /// assert_eq!(map.get_right(&2), None);
    /// assert_eq!(map.get_right(&3), None);
    /// ```
    pub fn get_right<Q>(&self, item: &Q) -> Option<&R>
    where
        L: Borrow<Q>,
        Q: Hash + Eq + PartialEq<L>,
    {
        let l_hash = make_hash::<_, S>(&self.hash_builder, item);
        let left_pairing: &MappingPair<L> = self.get_left_inner_with_hash(item, l_hash)?;
        let hash = left_pairing.hash?;
        match self
            .right_set
            .find(hash, hash_and_id(l_hash, left_pairing.id))
        {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    #[inline]
    fn get_left_inner_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<&MappingPair<L>>
    where
        L: Borrow<Q>,
        Q: Hash + Eq + PartialEq<L>,
    {
        self.left_set.find(hash, equivalent_key(item))
    }

    #[inline]
    fn get_right_inner_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<&MappingPair<R>>
    where
        R: Borrow<Q>,
        Q: Hash + Eq + PartialEq<R>,
    {
        self.right_set.find(hash, equivalent_key(item))
    }

    /// Returns an iterator over the items in the map
    ///
    /// Nope: The iterator will never yield the `Neither` variant of `OptionalPair` and will
    /// instead yield `None`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for op in map.iter() {
    ///     match op {
    ///         SomeBoth(l, r) => { println!("left: {l}, right: {r}"); }
    ///         SomeLeft(l) => { println!("just left: {l}"); }
    ///         SomeRight(r) => { println!("just right: {r}"); }
    ///         _ => { unreachable!("Never Neither"); }
    ///     }
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, L, R> {
        Iter(IterInner::FirstHalf {
            left_iter: self.left_set.iter(),
            map_ref: &self.right_set,
        })
    }

    /// Returns an iterator over the pairs in the map
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for (left, right) in map.iter_paired() {
    ///     println!("left: {left}, right: {right}");
    /// }
    /// ```
    pub fn iter_paired(&self) -> PairedIter<'_, L, R> {
        PairedIter {
            left_iter: self.left_set.iter(),
            map_ref: &self.right_set,
        }
    }

    /// Returns an iterator over the unpaired items in the map
    ///
    /// Nope: The iterator will never yield the `Neither` nor `SomeBoth` variants of
    /// `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for op in map.iter_unpaired() {
    ///     match op {
    ///         SomeLeft(l) => { println!("just left: {l}"); }
    ///         SomeRight(r) => { println!("just right: {r}"); }
    ///         _ => { unreachable!("Never Neither or SomeBoth"); }
    ///     }
    /// }
    /// ```
    pub fn iter_unpaired(&self) -> UnpairedIter<'_, L, R> {
        UnpairedIter {
            left_iter: self.left_set.iter(),
            right_iter: self.right_set.iter(),
        }
    }

    /// Returns an iterator over elements in the left set
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for left in map.iter_left() {
    ///     println!("left: {left}");
    /// }
    /// ```
    pub fn iter_left(&self) -> SingleIter<'_, L> {
        SingleIter {
            iter: self.left_set.iter(),
        }
    }

    /// Returns an iterator over elements in the right set
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let map: PartialCycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for right in map.iter_right() {
    ///     println!("right: {right}");
    /// }
    /// ```
    pub fn iter_right(&self) -> SingleIter<'_, R> {
        SingleIter {
            iter: self.right_set.iter(),
        }
    }

    // TODO: These are being temporary removed until the cursor API for HashTable is stablized.
    // The drain API had a clear bug in it and there was almost certainly a bug in the drain filter
    // iterator. These could be re-worked to function similarly to the `drain` method by buffering
    // everything before yielding them. I *strongly*
    /// Clears the map, returning all items as an iterator while keeping the backing memory
    /// allocated for reuse. If the returned iterator is dropped before being fully consumed, it
    /// drops the remaining pairs.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map = PartialCycleMap::new();
    /// map.insert(1, "1");
    /// map.insert(2, "2");
    /// let cap = map.capacity_left();
    ///
    /// for op in map.drain().take(1) {
    ///     match op {
    ///         SomeBoth(l,r) => {
    ///             assert!(l == 1 || l == 2);
    ///             assert!(r == "1" || r == "2");
    ///         }
    ///         _ => { unreachable!("Only pairs were inserted."); }
    ///     }
    /// }
    ///
    /// assert!(map.is_empty());
    /// assert_eq!(map.capacity_left(), cap);
    /// ```
    pub fn drain(&mut self) -> DrainIter<'_, L, R> {
        let mut values = Vec::with_capacity(self.left_set.len());
        values.extend(self.left_set.drain().map(|left| {
            match left
                .hash
                .and_then(|hash| self.right_set.find_entry(hash, just_id(left.id)).ok())
            {
                Some(entry) => EitherOrBoth::Both(left.value, entry.remove().0.value),
                None => EitherOrBoth::Left(left.value),
            }
        }));
        values.extend(
            self.right_set
                .drain()
                .map(|item| EitherOrBoth::Right(item.value)),
        );
        DrainIter(values.into_iter(), PhantomData)
    }

    /// Returns an iterator that removes and yields all items that evaluate to `true` in the given
    /// closure while keeping the backing memory allocated.
    ///
    /// If the closure returns `false`, or panics, the element remains in the map and will not be
    /// yielded.
    ///
    /// If the iterator is only partially consumed or not consumed at all, each of the remaining
    /// elements will still be subjected to the closure and removed and dropped if it returns true.
    ///
    /// It is unspecified how many more elements will be subjected to the closure if a panic occurs
    /// in the closure, or a panic occurs while dropping an element, or if the `DrainFilter` value
    /// is leaked.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..50).map(|i| (i,i.to_string())).collect();
    /// map.extend( (50..100).map(|i| SomeLeft(i)) );
    ///
    /// // Iterate over the map, remove all unpaired left items
    /// for op in map.drain_filter(|op| if let SomeLeft(_) = op { true } else { false }) {
    ///     assert!(op.get_left().is_some());
    ///     assert!(op.get_right().is_none());
    /// }
    ///
    /// assert_eq!(map.len_left(), 50);
    /// ```
    pub fn extract_if<F>(&mut self, mut f: F) -> ExtractIfIter<'_, L, R, F>
    where
        F: FnMut(EitherOrBoth<&L, &R>) -> bool,
    {
        let mut values = Vec::with_capacity(self.left_set.len());
        for left in self.left_set.iter() {
            match left
                .hash
                .and_then(|hash| self.right_set.find_entry(hash, just_id(left.id)).ok())
            {
                Some(entry) => {
                    if f(EitherOrBoth::Both(&left.value, &entry.get().value)) {
                        values.push((
                            left.id,
                            EitherOrBoth::Both(
                                make_hash(&self.hash_builder, &left.value),
                                make_hash(&self.hash_builder, &entry.get().value),
                            ),
                        ));
                    }
                }
                None => {
                    if f(EitherOrBoth::Left(&left.value)) {
                        values.push((
                            left.id,
                            EitherOrBoth::Left(make_hash(&self.hash_builder, &left.value)),
                        ));
                    }
                }
            };
        }
        for item in self.right_set.iter().filter(|item| item.hash.is_none()) {
            if f(EitherOrBoth::Right(&item.value)) {
                values.push((
                    item.id,
                    EitherOrBoth::Right(make_hash(&self.hash_builder, &item.value)),
                ));
            }
        }
        let mut to_drop = Vec::with_capacity(values.len());
        for (id, pair) in values {
            match pair {
                EitherOrBoth::Left(left) => {
                    let Ok(entry) = self.left_set.find_entry(left, just_id(id)) else {
                        unreachable!("TODO");
                    };
                    to_drop.push(EitherOrBoth::Left(entry.remove().0.value));
                }
                EitherOrBoth::Right(right) => {
                    let Ok(entry) = self.right_set.find_entry(right, just_id(id)) else {
                        unreachable!("TODO");
                    };
                    to_drop.push(EitherOrBoth::Right(entry.remove().0.value));
                }
                EitherOrBoth::Both(left, right) => {
                    let Ok(left) = self.left_set.find_entry(left, just_id(id)) else {
                        unreachable!("TODO");
                    };
                    let Ok(right) = self.right_set.find_entry(right, just_id(id)) else {
                        unreachable!("TODO");
                    };
                    to_drop.push(EitherOrBoth::Both(
                        left.remove().0.value,
                        right.remove().0.value,
                    ));
                }
            }
        }
        ExtractIfIter(to_drop.into_iter(), PhantomData)
    }

    /// Drops all pairs that cause the given closure to return `false`. Pairs are visited in an
    /// arbitary order.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..50).map(|i| (i,i.to_string())).collect();
    /// map.extend( (50..100).map(|i| SomeLeft(i)) );
    ///
    /// // Iterate over the map, remove all unpaired left items
    /// map.retain(|op| if let SomeLeft(_) = op { true } else { false });
    ///
    /// assert_eq!(map.len_left(), 50);
    /// ```
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(EitherOrBoth<&L, &R>) -> bool,
    {
        self.extract_if(f).for_each(drop);
    }

    /// Drops all pairs that cause the predicate to return `false` while keeping the backing memory
    /// allocated
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..50).map(|i| (i,i.to_string())).collect();
    /// map.extend( (50..100).map(|i| SomeLeft(i)) );
    ///
    /// // Iterate over the map, remove all pairs with an odd left item
    /// map.retain_paired(|l, r| l % 2 == 0);
    ///
    /// assert_eq!(map.len_left(), 75);
    /// ```
    pub fn retain_paired<F>(&mut self, mut f: F)
    where
        F: FnMut(&L, &R) -> bool,
    {
        self.left_set.retain(|left| {
            if let Some(hash) = left.hash {
                let Ok(entry) = self.right_set.find_entry(hash, just_id(left.id)) else {
                    todo!()
                };
                let do_remove = f(&left.value, &entry.get().value);
                if do_remove {
                    entry.remove();
                }
                do_remove
            } else {
                false
            }
        })
    }

    /// Drops all unpaired items that cause the predicate to return `false` while keeping the
    /// backing memory allocated
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{PartialCycleMap, OptionalPair::*};
    ///
    /// let mut map: PartialCycleMap<u64, String> = (0..50).map(|i| (i,i.to_string())).collect();
    /// map.extend( (50..100).map(|i| SomeLeft(i)) );
    ///
    /// // Iterate over the map, remove all unpaired odd left items
    /// map.retain_unpaired(|op| if let SomeLeft(l) = op { *l % 2 == 0 } else { true });
    ///
    /// assert_eq!(map.len_left(), 75);
    /// ```
    pub fn retain_unpaired<F>(&mut self, mut f: F)
    where
        F: FnMut(Either<&L, &R>) -> bool,
    {
        self.left_set
            .retain(|item| item.hash.is_none() && f(Either::Left(&item.value)));
        self.right_set
            .retain(|item| item.hash.is_none() && f(Either::Right(&item.value)));
    }

    /// Shrinks the capacity of the left set with a lower limit. It will drop down no lower than the
    /// supplied limit while maintaining the internal rules and possibly leaving some space in
    /// accordance with the resize policy.
    ///
    /// This function does nothing if the current capacity is smaller than the supplied minimum capacity.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map: PartialCycleMap<i32, i32> = PartialCycleMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity_left() >= 100);
    /// map.shrink_to_left(10);
    /// assert!(map.capacity_left() >= 10);
    /// map.shrink_to_left(0);
    /// assert!(map.capacity_left() >= 2);
    /// map.shrink_to_left(10);
    /// assert!(map.capacity_left() >= 2);
    /// ```
    pub fn shrink_to_left(&mut self, min_capacity: usize) {
        self.left_set
            .shrink_to(min_capacity, make_hasher(&self.hash_builder));
    }

    /// Shrinks the capacity of the right set with a lower limit. It will drop down no lower than the
    /// supplied limit while maintaining the internal rules and possibly leaving some space in
    /// accordance with the resize policy.
    ///
    /// This function does nothing if the current capacity is smaller than the supplied minimum capacity.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map: PartialCycleMap<i32, i32> = PartialCycleMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity_right() >= 100);
    /// map.shrink_to_left(10);
    /// assert!(map.capacity_right() >= 10);
    /// map.shrink_to_left(0);
    /// assert!(map.capacity_right() >= 2);
    /// map.shrink_to_left(10);
    /// assert!(map.capacity_right() >= 2);
    /// ```
    pub fn shrink_to_right(&mut self, min_capacity: usize) {
        self.right_set
            .shrink_to(min_capacity, make_hasher(&self.hash_builder));
    }

    /// Shrinks the capacity of the map as much as possible. It will drop down as much as possible
    /// while maintaining the internal rules and possibly leaving some space in accordance with the
    /// resize policy.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map: PartialCycleMap<i32, i32> = PartialCycleMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity_left() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity_left() >= 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.left_set
            .shrink_to(self.len_left(), make_hasher(&self.hash_builder));
        self.right_set
            .shrink_to(self.len_right(), make_hasher(&self.hash_builder));
    }

    /// Reserves capacity for at least additional more elements to be inserted in the HashMap. The
    /// collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    /// Panics if the new allocation size overflows usize.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map: PartialCycleMap<&str, i32> = PartialCycleMap::new();
    /// map.reserve_left(10);
    /// ```
    pub fn reserve_left(&mut self, additional: usize) {
        self.left_set
            .reserve(additional, make_hasher(&self.hash_builder));
    }

    /// Reserves capacity for at least additional more elements to be inserted in the HashMap. The
    /// collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    /// Panics if the new allocation size overflows usize.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map: PartialCycleMap<&str, i32> = PartialCycleMap::new();
    /// map.reserve_right(10);
    /// ```
    pub fn reserve_right(&mut self, additional: usize) {
        self.right_set
            .reserve(additional, make_hasher(&self.hash_builder));
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted in the given
    /// `HashMap<K,V>`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Errors
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map: PartialCycleMap<&str, isize> = PartialCycleMap::new();
    /// map.try_reserve_left(10).expect("why is the test harness OMGing on 10 bytes?");
    /// ```
    pub fn try_reserve_left(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.left_set
            .try_reserve(additional, make_hasher(&self.hash_builder))?;
        Ok(())
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted in the given
    /// `HashMap<K,V>`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Errors
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let mut map: PartialCycleMap<&str, isize> = PartialCycleMap::new();
    /// map.try_reserve_right(10).expect("why is the test harness OMGing on 10 bytes?");
    /// ```
    pub fn try_reserve_right(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.right_set
            .try_reserve(additional, make_hasher(&self.hash_builder))?;
        Ok(())
    }
}

impl<L, R, S> Clone for PartialCycleMap<L, R, S>
where
    L: Eq + Hash + Clone,
    R: Eq + Hash + Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            left_set: self.left_set.clone(),
            right_set: self.right_set.clone(),
            counter: self.counter,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<L, R, S> Default for PartialCycleMap<L, R, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(Default::default())
    }
}

impl<L, R, S> fmt::Debug for PartialCycleMap<L, R, S>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_set().entries(self.iter()).finish()
    }
}

impl<L, R, S> PartialEq<PartialCycleMap<L, R, S>> for PartialCycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len_left() != other.len_left() && self.len_right() != other.len_right() {
            return false;
        }
        self.iter().all(|op| match op {
            EitherOrBoth::Left(l) => other.get_right(l).is_none(),
            EitherOrBoth::Right(r) => other.get_left(r).is_none(),
            EitherOrBoth::Both(l, r) => other.are_paired(l, r),
        })
    }
}

impl<L, R, S> Eq for PartialCycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
}

impl<L, R, S> PartialCycleMap<L, R, S> {
    /// Creates a `PartialCycleMap`and that uses the given hasher.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed to allow
    /// `PartialCycleMap`s to be resistant to attacks that cause many collisions and very poor
    /// performance. Setting it manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for the CycleMap to be
    /// useful, see its documentation for details.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = PartialCycleMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, "1");
    /// ```
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            counter: 0,
            left_set: HashTable::new(),
            right_set: HashTable::new(),
        }
    }

    /// Creates a `PartialCycleMap` with inner sets that have at least the specified capacity, and that
    /// uses the given hasher.
    ///
    /// The map will be able to hold at least `capacity` many pairs before reallocating.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed to allow
    /// `PartialCycleMap`s to be resistant to attacks that cause many collisions and very poor
    /// performance. Setting it manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for the CycleMap to be
    /// useful, see its documentation for details.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = PartialCycleMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, "1");
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            hash_builder,
            counter: 0,
            left_set: HashTable::with_capacity(capacity),
            right_set: HashTable::with_capacity(capacity),
        }
    }

    /// Returns a reference to the [`BuildHasher`] used by the map
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of items that the left set can hold without reallocation.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let map: PartialCycleMap<u64, String> = PartialCycleMap::with_capacity(100);
    /// assert!(map.capacity_left() >= 100);
    /// ```
    pub fn capacity_left(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.capacity()
    }

    /// Returns the number of items that the right set can hold without reallocation.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    /// let map: PartialCycleMap<u64, String> = PartialCycleMap::with_capacity(100);
    /// assert!(map.capacity_right() >= 100);
    /// ```
    pub fn capacity_right(&self) -> usize {
        // The size of the sets is always equal
        self.right_set.capacity()
    }

    /// Returns the len of the inner sets (same between sets)
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// assert_eq!(map.len_left(), 0);
    /// map.insert(1, "1");
    /// map.insert_left(2);
    /// assert_eq!(map.len_left(), 2);
    /// ```
    pub fn len_left(&self) -> usize {
        self.left_set.len()
    }

    /// Returns the len of the inner sets (same between sets)
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// assert_eq!(map.len_right(), 0);
    /// map.insert(1, "1");
    /// map.insert_right("2");
    /// assert_eq!(map.len_right(), 2);
    /// ```
    pub fn len_right(&self) -> usize {
        self.right_set.len()
    }

    /// Returns true if no items are in the map and false otherwise
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// assert!(map.is_empty());
    /// map.insert(1, "1");
    /// assert_eq!(map.len_left(), 1);
    /// assert!(!map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len_left() + self.len_right() == 0
    }

    /// Removes all items for the map while keeping the backing memory allocated for reuse.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::PartialCycleMap;
    ///
    /// let mut map = PartialCycleMap::new();
    /// assert!(map.is_empty());
    /// map.insert(1, "a");
    /// assert!(!map.is_empty());
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.left_set.clear();
        self.right_set.clear();
    }
}

impl<L, R, S> Extend<(L, R)> for PartialCycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (L, R)>>(&mut self, iter: T) {
        for (l, r) in iter {
            self.insert(l, r);
        }
    }
}

impl<L, R, S> Extend<OptionalPair<L, R>> for PartialCycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    #[inline]
    fn extend<T: IntoIterator<Item = OptionalPair<L, R>>>(&mut self, iter: T) {
        for op in iter {
            match op {
                Neither => {}
                SomeLeft(l) => {
                    self.insert_left(l);
                }
                SomeRight(r) => {
                    self.insert_right(r);
                }
                SomeBoth(l, r) => {
                    self.insert(l, r);
                }
            }
        }
    }
}

impl<L, R> FromIterator<(L, R)> for PartialCycleMap<L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = (L, R)>>(iter: T) -> Self {
        let mut digest = PartialCycleMap::default();
        digest.extend(iter);
        digest
    }
}

impl<L, R> FromIterator<OptionalPair<L, R>> for PartialCycleMap<L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = OptionalPair<L, R>>>(iter: T) -> Self {
        let mut digest = PartialCycleMap::default();
        digest.extend(iter);
        digest
    }
}

/// An iterator over the entry items of a `PartialCycleMap`.
pub struct Iter<'a, L, R>(IterInner<'a, L, R>);

enum IterInner<'a, L, R> {
    FirstHalf {
        left_iter: hash_table::Iter<'a, MappingPair<L>>,
        map_ref: &'a HashTable<MappingPair<R>>,
    },
    Rest {
        right_iter: hash_table::Iter<'a, MappingPair<R>>,
    },
}

impl<L, R> Clone for Iter<'_, L, R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<L, R> Clone for IterInner<'_, L, R> {
    fn clone(&self) -> Self {
        match self {
            Self::FirstHalf { left_iter, map_ref } => Self::FirstHalf {
                left_iter: left_iter.clone(),
                map_ref,
            },
            Self::Rest { right_iter } => Self::Rest {
                right_iter: right_iter.clone(),
            },
        }
    }
}

impl<L, R> fmt::Debug for Iter<'_, L, R>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(Self::clone(self)).finish()
    }
}

impl<'a, L, R> Iterator for Iter<'a, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    type Item = EitherOrBoth<&'a L, &'a R>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            IterInner::FirstHalf { left_iter, map_ref } => {
                let Some(left_item) = left_iter.next() else {
                    self.0 = IterInner::Rest {
                        right_iter: map_ref.iter(),
                    };
                    return self.next();
                };
                let opt_right = left_item
                    .hash
                    .map(|hash| &map_ref.find(hash, just_id(left_item.id)).unwrap().value);
                let digest = match opt_right {
                    Some(right) => EitherOrBoth::Both(&left_item.value, right),
                    None => EitherOrBoth::Left(&left_item.value),
                };
                Some(digest)
            }
            IterInner::Rest { right_iter } => {
                // NOTE: This could be expressed as a filter, but that adds an extra generic, so
                // this will suffice.
                loop {
                    let item = right_iter.next()?;
                    if item.hash.is_none() {
                        return Some(EitherOrBoth::Right(&item.value));
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<L, R> ExactSizeIterator for Iter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn len(&self) -> usize {
        self.clone().count()
    }
}

impl<L, R> FusedIterator for Iter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
}

/// An iterator over the paired entry of a `PartialCycleMap`.
pub struct PairedIter<'a, L, R> {
    left_iter: hash_table::Iter<'a, MappingPair<L>>,
    map_ref: &'a HashTable<MappingPair<R>>,
}

impl<L, R> Clone for PairedIter<'_, L, R> {
    fn clone(&self) -> Self {
        Self {
            left_iter: self.left_iter.clone(),
            map_ref: self.map_ref,
        }
    }
}

impl<L, R> fmt::Debug for PairedIter<'_, L, R>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, L, R> Iterator for PairedIter<'a, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    type Item = (&'a L, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let left_item = self.left_iter.next()?;
            if let Some(hash) = left_item.hash {
                let right_item = self.map_ref.find(hash, just_id(left_item.id)).unwrap();
                return Some((&left_item.value, &right_item.value));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}

impl<L, R> ExactSizeIterator for PairedIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn len(&self) -> usize {
        self.clone().count()
    }
}

impl<L, R> FusedIterator for PairedIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
}

/// An iterator over the entry pairs of a `PartialCycleMap`.
pub struct UnpairedIter<'a, L, R> {
    left_iter: hash_table::Iter<'a, MappingPair<L>>,
    right_iter: hash_table::Iter<'a, MappingPair<R>>,
}

impl<L, R> Clone for UnpairedIter<'_, L, R> {
    fn clone(&self) -> Self {
        Self {
            left_iter: self.left_iter.clone(),
            right_iter: self.right_iter.clone(),
        }
    }
}

impl<L, R> fmt::Debug for UnpairedIter<'_, L, R>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, L, R> Iterator for UnpairedIter<'a, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    type Item = Either<&'a L, &'a R>;

    fn next(&mut self) -> Option<Self::Item> {
        for left_item in self.left_iter.by_ref() {
            // Ignore all paired items
            if left_item.hash.is_some() {
                continue;
            }
            return Some(Either::Left(&left_item.value));
        }
        for right_item in self.right_iter.by_ref() {
            // Ignore all paired items
            if right_item.hash.is_some() {
                continue;
            }
            return Some(Either::Right(&right_item.value));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}

impl<L, R> ExactSizeIterator for UnpairedIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn len(&self) -> usize {
        self.clone().count()
    }
}

impl<L, R> FusedIterator for UnpairedIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
}

/// An iterator over the elements of an inner set of a `PartialCycleMap`.
pub struct SingleIter<'a, T> {
    iter: hash_table::Iter<'a, MappingPair<T>>,
}

impl<T> Clone for SingleIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<T> fmt::Debug for SingleIter<'_, T>
where
    T: Hash + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, T> Iterator for SingleIter<'a, T>
where
    T: 'a + Hash + Eq,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| &item.value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> ExactSizeIterator for SingleIter<'_, T>
where
    T: Hash + Eq,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T> FusedIterator for SingleIter<'_, T> where T: Hash + Eq {}

// NOTE: The lifetime is not currently needed because we buffer all changes, but, when the cursor
// API is stablized, the changes will not be buffered and we will need the lifetime. This prevents
// a (minor) breaking change.
/// An iterator that removes all items and pairs of items from the backing `PartialCycleMap` and
/// yeilds them.
#[allow(missing_debug_implementations)]
pub struct DrainIter<'a, L, R>(vec::IntoIter<EitherOrBoth<L, R>>, PhantomData<&'a ()>);

impl<L, R> Iterator for DrainIter<'_, L, R> {
    type Item = EitherOrBoth<L, R>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

// NOTE: The lifetime and `F` (predicate) is not currently needed because we buffer all changes, but, when the cursor
// API is stablized, the changes will not be buffered and we will need the lifetime. This prevents
// a (minor) breaking change.
/// An iterator similar to [`DrainIter`] but that applies a filter to what is removed.
#[allow(missing_debug_implementations)]
pub struct ExtractIfIter<'a, L, R, F>(vec::IntoIter<EitherOrBoth<L, R>>, PhantomData<&'a F>);

impl<L, R, F> Iterator for ExtractIfIter<'_, L, R, F> {
    type Item = EitherOrBoth<L, R>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> MappingPair<T> {
    // Consumes the pair and returns the held `T`
    pub(crate) fn extract(self) -> T {
        self.value
    }
}

impl<T: Hash> Hash for MappingPair<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl<T: PartialEq> PartialEq for MappingPair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) && self.value.eq(&other.value)
    }
}

impl<T: PartialEq> PartialEq<T> for MappingPair<T> {
    fn eq(&self, other: &T) -> bool {
        self.value.eq(other)
    }
}

impl<T: Eq> Eq for MappingPair<T> {}

impl<T: Clone> Clone for MappingPair<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            hash: self.hash,
            id: self.id,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for MappingPair<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MappingPair {{ value: {:?}, hash: {:?}, id: {} }}",
            self.value, self.hash, self.id
        )
    }
}
