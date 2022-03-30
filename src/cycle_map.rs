use core::{borrow::Borrow, mem};
use std::{
    borrow::BorrowMut,
    collections::{hash_map::RandomState, HashMap},
    default::Default,
    fmt,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    marker::PhantomData,
};

use hashbrown::{
    hash_map::DefaultHashBuilder,
    raw::{RawDrain, RawIntoIter, RawIter, RawTable},
};

use crate::optional_pair::OptionalPair;

fn equivalent_key<Q: PartialEq + ?Sized>(k: &Q) -> impl Fn(&MappingPair<Q>) -> bool + '_ {
    move |x| k.eq(&x.value)
}

fn does_map<'a, P, Q>(
    value: &'a P,
    map_ref: &'a RawTable<MappingPair<P>>,
) -> impl Fn(&MappingPair<Q>) -> bool + 'a
where
    P: Hash + PartialEq + Eq,
    Q: Hash + PartialEq + Eq + ?Sized,
{
    move |pair| map_ref.get(pair.hash, equivalent_key(value)).is_some()
}

fn make_hash<T, S>(hash_builder: &S, val: &T) -> u64
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

fn make_hasher<T, S>(hash_builder: &S) -> impl Fn(&T) -> u64 + '_
where
    T: Hash,
    S: BuildHasher,
{
    move |val| make_hash::<T, S>(hash_builder, val)
}

// Contains a value and the hash of the item that the value maps to.
pub(crate) struct MappingPair<T> {
    pub(crate) value: T,
    pub(crate) hash: u64,
}

impl<T> MappingPair<T> {
    // Consumes the pair and returns the held `T`
    pub fn extract(self) -> T {
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
        self.value.eq(&other.value)
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
            hash: self.hash.clone(),
        }
    }
}

impl<T: Debug> fmt::Debug for MappingPair<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MappingPair {{ value: {:?}, hash: {} }}",
            self.value, self.hash
        )
    }
}

/// A hash map that supports bidirection searches.
///
/// [`CycleMap`] bijectively maps two sets of elements, i.e. every element always
/// has a "companion". It does this while maintaining the same complexitity for "gets"
/// as a traditional [`HashMap`] and while only keeping a single copy of each element.
///
/// It is implemented using two sets, a "left" and "right" set. On insert, the given pair
/// of items is split. The left item is stored in the left set with the hash of the right item,
/// likewise for the right item. As such, both the left and right types need to implement [`Eq`]
/// and [`Hash`], and as with other hashed collections, it is a logic error for an item to be
/// modified in such a way that the item's hash or its equality, as changes while it is in the bag.
///
/// Sorting values like this allows for look up on pair with a standard HashMap and makes resizes
/// faster but is not with a cost. When inserting a new pair of elements, there is potentail for
/// collision. This collision should be excendingly rare and can only happen upon inserting new
/// elements. You can read more about what causes collisions [here]("").
pub struct CycleMap<L, R, St = DefaultHashBuilder> {
    pub(crate) hash_builder: St,
    left_set: RawTable<MappingPair<L>>,
    right_set: RawTable<MappingPair<R>>,
}

impl<L, R> CycleMap<L, R, DefaultHashBuilder> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<L, R, S> CycleMap<L, R, S>
where
    L: Eq + Hash,
    R: Eq + Hash,
    S: BuildHasher,
{
    /// Adds a pair of items to the map.
    ///
    /// Should the left element be equal to another left element, the old pair is removed and
    /// returned. The same goes for the right element.
    ///
    /// There is a chance for collision here. Collision occurs when the map contains elements with
    /// identical hashes as the given left and right elements, and they are mapped to each other.
    /// In such a case, the old pair is returned and the new pair is inserted.
    pub fn insert(&mut self, left: L, right: R) -> OptionalPair<L, R> {
        todo!()
    }

    /// Determines if two items are mapped to one another
    ///
    /// Returns false if either item isn't found it its associated list.
    pub fn are_mapped(&self, left: &L, right: &R) -> bool {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let opt_left = self.left_set.get(l_hash, equivalent_key(left));
        let opt_right = self.right_set.get(r_hash, equivalent_key(right));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => l_hash == right.hash && r_hash == left.hash,
            _ => false,
        }
    }

    /// Removes the given item from the left set and its associated item from the right set
    pub fn remove_via_left(&mut self, item: &L) -> Option<(L, R)> {
        // Be careful here... removing an element too early can cause issues
        let right_pairing: MappingPair<R> = unsafe { self.take_right(item)? };
        let l_hash = make_hash::<L, S>(&self.hash_builder, item);
        let left_pairing: MappingPair<L> =
            self.left_set.remove_entry(l_hash, equivalent_key(item))?;
        Some((left_pairing.extract(), right_pairing.extract()))
    }

    /// Removes the given item from the right set and its associated item from the left set
    pub fn remove_via_right(&mut self, item: &R) -> Option<(L, R)> {
        // Be careful here... removing an element too early can cause issues
        let left_pairing: MappingPair<L> = unsafe { self.take_left(item)? };
        let r_hash = make_hash::<R, S>(&self.hash_builder, item);
        let right_pairing: MappingPair<R> =
            self.right_set.remove_entry(r_hash, equivalent_key(item))?;
        Some((left_pairing.extract(), right_pairing.extract()))
    }

    /// Removes a pair of items only if they are mapped together and returns the pair
    pub fn remove(&mut self, left: &L, right: &R) -> Option<(L, R)> {
        let l_hash = make_hash::<L, S>(&self.hash_builder, left);
        let r_hash = make_hash::<R, S>(&self.hash_builder, right);
        let left_pairing: MappingPair<L> =
            self.left_set.remove_entry(l_hash, equivalent_key(left))?;
        let right_pairing: MappingPair<R> =
            self.right_set.remove_entry(r_hash, equivalent_key(right))?;
        Some((left_pairing.extract(), right_pairing.extract()))
    }

    /// Swaps an item in the left set with another item, remaps the old item's associated right
    /// item, and returns the old left item
    pub fn swap_left(&mut self, new: L, old: &L) -> Option<L> {
        let old_l_hash = make_hash::<L, S>(&self.hash_builder, old);
        let left_pairing: MappingPair<L> = self
            .left_set
            .remove_entry(old_l_hash, equivalent_key(old))?;
        let r_pairing: &mut MappingPair<R> = self
            .right_set
            .get_mut(
                left_pairing.hash,
                does_map(&left_pairing.value, &self.left_set),
            )
            .unwrap();
        let new_l_hash = make_hash::<L, S>(&self.hash_builder, &new);
        r_pairing.hash = new_l_hash;
        self.left_set.insert(
            new_l_hash,
            MappingPair {
                value: new,
                hash: left_pairing.hash,
            },
            make_hasher::<MappingPair<L>, S>(&self.hash_builder),
        );
        Some(left_pairing.extract())
    }

    /// Does what [`swap_left`] does, but fails to swap if the old item isn't mapped to the given
    /// right item.
    ///
    /// [`swap_left`]: struct.CycleMap.html#method.swap_left
    pub fn swap_left_checked(&mut self, new: L, old: &L, expected: &R) -> Option<L> {
        todo!()
    }

    /// Does what [`swap_left`] does, but inserts a new pair if the old left item isn't in the map
    ///
    /// [`swap_left`]: struct.CycleMap.html#method.swap_left
    pub fn swap_left_or_insert(&mut self, new: L, old: &L, to_insert: R) -> Option<L> {
        todo!()
    }

    /// Swaps an item in the right set with another item, remaps the old item's associated left
    /// item, and returns the old right item
    pub fn swap_right(&mut self, new: R, old: &R) -> Option<R> {
        let old_r_hash = make_hash::<R, S>(&self.hash_builder, old);
        let right_pairing: MappingPair<R> = self
            .right_set
            .remove_entry(old_r_hash, equivalent_key(old))?;
        let l_pairing: &mut MappingPair<L> = self
            .left_set
            .get_mut(
                right_pairing.hash,
                does_map(&right_pairing.value, &self.right_set),
            )
            .unwrap();
        let new_r_hash = make_hash::<R, S>(&self.hash_builder, &new);
        l_pairing.hash = new_r_hash;
        self.right_set.insert(
            new_r_hash,
            MappingPair {
                value: new,
                hash: right_pairing.hash,
            },
            make_hasher::<MappingPair<R>, S>(&self.hash_builder),
        );
        Some(right_pairing.extract())
    }

    /// Does what [`swap_right`] does, but fails to swap if the old item isn't mapped to the given
    /// left item.
    ///
    /// [`swap_right`]: struct.CycleMap.html#method.swap_right
    pub fn swap_right_checked(&mut self, new: R, old: &R, expected: &L) -> Option<R> {
        todo!()
    }

    /// Does what [`swap_right`] does, but inserts a new pair if the old right item isn't in the map
    ///
    /// [`swap_right`]: struct.CycleMap.html#method.swap_right
    pub fn swap_right_or_insert(&mut self, new: R, old: &R, to_insert: L) -> Option<R> {
        todo!()
    }

    /// Gets a reference to an item in the left set using an item in the right set.
    pub fn get_left(&self, item: &R) -> Option<&L> {
        let right_pairing: &MappingPair<R> = self.get_right_inner(item)?;
        match self.left_set.get(
            right_pairing.hash,
            does_map(&right_pairing.value, &self.right_set),
        ) {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    /// Gets a reference to an item in the right set using an item in the left set.
    pub fn get_right(&self, item: &L) -> Option<&R> {
        let left_pairing: &MappingPair<L> = self.get_left_inner(item)?;
        match self.right_set.get(
            left_pairing.hash,
            does_map(&left_pairing.value, &self.left_set),
        ) {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    #[inline]
    fn get_left_inner(&self, item: &L) -> Option<&MappingPair<L>> {
        let hash = make_hash::<L, S>(&self.hash_builder, item);
        self.left_set.get(hash, equivalent_key(item))
    }

    #[inline]
    fn get_right_inner(&self, item: &R) -> Option<&MappingPair<R>> {
        let hash = make_hash::<R, S>(&self.hash_builder, item);
        self.right_set.get(hash, equivalent_key(item))
    }

    /// Takes an item from the left set and returns it (if it exists).
    ///
    /// This method is unsafe since removing the item break a cycle in the map.
    /// Ensure that any element you remove this way has its corresponding item removed too.
    pub(crate) unsafe fn take_left(&mut self, item: &R) -> Option<MappingPair<L>> {
        let r_hash = make_hash::<R, S>(&self.hash_builder, item);
        let right_pairing: &MappingPair<R> = self.right_set.get(r_hash, equivalent_key(item))?;
        self.left_set.remove_entry(
            right_pairing.hash,
            does_map(&right_pairing.value, &self.right_set),
        )
    }

    /// Takes an item from the right set and returns it (if it exists).
    ///
    /// This method is unsafe since removing the item break a cycle in the map.
    /// Ensure that any element you remove this way has its corresponding item removed too.
    pub(crate) unsafe fn take_right(&mut self, item: &L) -> Option<MappingPair<R>> {
        let l_hash = make_hash::<L, S>(&self.hash_builder, item);
        let left_pairing: &MappingPair<L> = self.left_set.get(l_hash, equivalent_key(item))?;
        self.right_set.remove_entry(
            left_pairing.hash,
            does_map(&left_pairing.value, &self.left_set),
        )
    }

    pub fn iter(&self) -> Iter<'_, L, R, S> {
        Iter {
            left_iter: unsafe { self.left_set.iter() },
            map_ref: self,
        }
    }

    pub fn iter_left(&self) -> SingleIter<'_, L> {
        SingleIter {
            iter: unsafe { self.left_set.iter() },
            marker: PhantomData,
        }
    }

    pub fn iter_right(&self) -> SingleIter<'_, R> {
        SingleIter {
            iter: unsafe { self.right_set.iter() },
            marker: PhantomData,
        }
    }

    /*
     * TODO: Revisit these...
    pub fn drain(&mut self) -> Drain<'_, L, R, S> {
        Drain {
            left_iter: self.left_set.drain(),
            right_ref: &mut self,
        }
    }

    pub fn drain_filter<F>(&mut self, f: F) -> DrainFilter<'_, L, R, S, F>
    where
        F: FnMut(&L, &R) -> bool,
    {
        DrainFilter {
            f,
            inner: DrainFilterInner {
                left_iter: unsafe { self.left_set.iter() },
                map_ref: &mut self,
            },
        }
    }
    */
}

impl<L, R, S> Default for CycleMap<L, R, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(Default::default())
    }
}

impl<L, R, S> CycleMap<L, R, S> {
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            left_set: RawTable::new(),
            right_set: RawTable::new(),
        }
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            hash_builder,
            left_set: RawTable::new(),
            right_set: RawTable::new(),
        }
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    pub fn capacity(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.capacity()
    }

    fn raw_capacity(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.buckets()
    }

    pub fn len(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&L, &R) -> bool,
    {
        todo!()
    }

    pub fn clear(&mut self) {
        self.left_set.clear();
        self.right_set.clear();
    }
}

/// An iterator over the entry pairs of a `CycleMap`.
pub struct Iter<'a, L, R, S> {
    left_iter: RawIter<MappingPair<L>>,
    map_ref: &'a CycleMap<L, R, S>,
}

impl<L, R, S> Clone for Iter<'_, L, R, S> {
    fn clone(&self) -> Self {
        Self {
            left_iter: self.left_iter.clone(),
            map_ref: self.map_ref,
        }
    }
}

impl<L, R, S> fmt::Debug for Iter<'_, L, R, S>
where
    L: Hash + Eq + Debug,
    R: Hash + Eq + Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, L, R, S> Iterator for Iter<'a, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    type Item = (&'a L, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        match self.left_iter.next() {
            Some(l) => unsafe {
                let left = &l.as_ref().value;
                let right = self.map_ref.get_right(&left).unwrap();
                Some((left, right))
            },
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}

impl<L, R, S> ExactSizeIterator for Iter<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    fn len(&self) -> usize {
        self.left_iter.len()
    }
}

impl<L, R, S> FusedIterator for Iter<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
}

/// An iterator over the left elements of a `CycleMap`.
pub struct SingleIter<'a, T> {
    iter: RawIter<MappingPair<T>>,
    marker: PhantomData<&'a T>,
}

impl<T> Clone for SingleIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for SingleIter<'_, T>
where
    T: Hash + Eq + Debug,
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
        match self.iter.next() {
            Some(item) => {
                let val = unsafe { &item.as_ref().value };
                Some(val)
            }
            None => None,
        }
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

/*
 * TODO: Revisit these too...
/// An iterator over the entry pairs of a `CycleMap`.
pub struct Drain<'a, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    left_iter: RawDrain<'a, MappingPair<L>>,
    map_ref: &'a mut CycleMap<L, R, S>,
}

impl<L, R, S> Drain<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    /// Returns a iterator of references over the remaining items.
    pub(super) fn iter(&self) -> Iter<'_, L, R, S> {
        Iter {
            left_iter: self.left_iter.iter(),
            map_ref: self.map_ref,
        }
    }
}

impl<'a, L, R, S> Iterator for Drain<'a, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    type Item = (L, R);

    fn next(&mut self) -> Option<Self::Item> {
        match self.left_iter.next() {
            Some(l) => unsafe {
                let left = l.extract();
                let right = self.map_ref.take_right(&left).unwrap().extract();
                Some((left, right))
            },
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}
impl<L, R, S> ExactSizeIterator for Drain<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    fn len(&self) -> usize {
        self.left_iter.len()
    }
}
impl<L, R, S> FusedIterator for Drain<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
}

impl<L, R, S> fmt::Debug for Drain<'_, L, R, S>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

pub struct DrainFilter<'a, L, R, S, F>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
    F: FnMut(&L, &R) -> bool,
{
    f: F,
    inner: DrainFilterInner<'a, L, R, S>,
}

impl<'a, L, R, S, F> Drop for DrainFilter<'a, L, R, S, F>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
    F: FnMut(&L, &R) -> bool,
{
    fn drop(&mut self) {
        while let Some(item) = self.next() {
            let guard = ConsumeAllOnDrop(self);
            drop(item);
            mem::forget(guard);
        }
    }
}

pub(super) struct ConsumeAllOnDrop<'a, T: Iterator>(pub &'a mut T);

impl<T: Iterator> Drop for ConsumeAllOnDrop<'_, T> {
    fn drop(&mut self) {
        self.0.for_each(drop)
    }
}

impl<L, R, S, F> Iterator for DrainFilter<'_, L, R, S, F>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
    F: FnMut(&L, &R) -> bool,
{
    type Item = (L, R);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.left_iter.size_hint().1)
    }
}

impl<L, R, S, F> FusedIterator for DrainFilter<'_, L, R, S, F>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
    F: FnMut(&L, &R) -> bool,
{
}

pub(super) struct DrainFilterInner<'a, L, R, S> {
    pub left_iter: RawIter<MappingPair<L>>,
    pub map_ref: &'a mut CycleMap<L, R, S>,
}

impl<L, R, S> DrainFilterInner<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    pub(super) fn next<F>(&mut self, f: &mut F) -> Option<(L, R)>
    where
        F: FnMut(&L, &R) -> bool,
    {
        while let Some(item) = self.left_iter.next() {
            let left = unsafe { &item.as_ref().value };
            let right = self.map_ref.get_right(&left).unwrap();
            if f(left, right) {
                return self.map_ref.remove(left, right);
            }
        }
        None
    }
}
*/
