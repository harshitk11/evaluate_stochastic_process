# NetLogo-Forest-Fire-evolution

This repository contains code to model the evolution of forest fires using the NetLogo multi-agent modelling framework.

To run the [code](netlogo_models/forest_fire_evolution.nlogo), you'll need the [NetLogo agent](http://ccl.northwestern.edu/netlogo/models/community/Agent-Based%20Model). You can also get a high-level description of the model from [this image](fire_evolution_dynamics_explained.PNG).

For generating the dataset from scratch and/or to visualize the results reported in Figure 3 of the paper (characterization of forest fire as a stochastic process), please see [pyNetLogo_script.py](pyNetLogo_script.py)

## Installing NetLogo

The codebase is tested with NetLogo v6.0. To set it up:

1. Download and unpack NetLogo:
    ```bash
    wget http://ccl.northwestern.edu/netlogo/6.0.0/NetLogo-6.0-64.tgz
    tar -xzf NetLogo-6.0-64.tgz
    rm NetLogo-6.0-64.tgz
    ```

2. Rename the extracted directory to "NetLogo6" (removing space):
   This step prevents downstream file access exceptions.

## Integration with pyNetLogo

To integrate NetLogo with Python (on Linux), follow the provided [guidelines](https://pynetlogo.readthedocs.io/en/latest/install.html). Here are some additional steps to prevent potential bugsåå:

1. Ensure Java is installed:
    - Check using: `which java`
    - Set `$JAVA_HOME`: 
        ```bash
        echo "export JAVA_HOME=/path/to/java" >> ~/.bashrc
        ```
    - If not installed, get Java for Ubuntu [here](https://ubuntu.com/tutorials/install-jre#2-installing-openjdk-jre).

2. Ensure NetLogo 6.0.0 is set up:
    - Make necessary modifications as detailed [here](https://github.com/NetLogo/NetLogo/issues/1361) before running `netlogo-headless.sh`.

3. Address memory constraints:
    - Modify `netlogo-headless.sh` to allocate more RAM to prevent `java.lang.OutOfMemoryError`.

4. Set up the environment:
    ```bash
    conda env create --prefix ./pynetlogo_env -f  environment.yml
    ```

5. When using pyNetLogo:
    - Ensure you specify the NetLogo version as '6.0' when establishing the NetLogoLink.

## Fixing pyNetLogo Bug

1. Navigate to the pyNetLogo core:
    ```
    pynetlogo_env/lib/python3.6/site-packages/pyNetLogo/core.py
    ```

2. Replace the `patch_report` and `patch_set` methods with the provided code. 

```python
def patch_report(self, attribute):
        """Return patch attributes from NetLogo

        Returns a pandas DataFrame with same dimensions as the NetLogo world,
        with column labels and row indices following pxcor and pycor patch
        coordinates. Values of the dataframe correspond to patch attributes.

        Parameters
        ----------
        attribute : str
            Valid NetLogo patch attribute

        Returns
        -------
        pandas DataFrame
            DataFrame containing patch attributes

        Raises
        ------
        NetLogoException
            If a LogoException or CompilerException is raised by NetLogo

        """

        try:
            extents = self.link.report('(list min-pxcor max-pxcor \
                                         min-pycor max-pycor)')
            extents = self._cast_results(extents).astype(int)

            results_df = pd.DataFrame(index=range(extents[2],
                                                  extents[3]+1, 1),
                                      columns=range(extents[0],
                                                    extents[1]+1, 1))
            results_df.sort_index(ascending=False, inplace=True)

            if self.netlogo_version == '5':
                resultsvec = self.link.report('map [[{0}] of ?] \
                                               sort patches'.format(attribute))
            else:
                resultsvec = self.link.report('map [[p] -> [{0}] of p] \
                                               sort patches'.format(attribute))    
            resultsvec = self._cast_results(resultsvec)
            results_df.iloc[:, :] = resultsvec.reshape(results_df.shape)

            return results_df

        except jpype.JException as ex:
            print(ex.stacktrace())
            raise NetLogoException(str(ex))

    def patch_set(self, attribute, data):
        """Set patch attributes in NetLogo

        Inverse of the `patch_report` method. Sets a patch attribute using
        values from a pandas DataFrame of same dimensions as the NetLogo world.

        Parameters
        ----------
        attribute : str
            Valid NetLogo patch attribute
        data : Pandas DataFrame
            DataFrame with same dimensions as NetLogo world

        Raises
        ------
        NetLogoException
            If a LogoException or CompilerException is raised by NetLogo

        """

        try:
            np.set_printoptions(threshold=np.prod(data.shape))
            datalist = '['+str(data.values.flatten()).strip('[ ')
            datalist = ' '.join(datalist.split())
            if self.netlogo_version == '5':
                command = '(foreach map [[pxcor] of ?] \
                            sort patches map [[pycor] of ?] \
                            sort patches {0} [ask patch ?1 ?2 \
                            [set {1} ?3]])'.format(datalist, attribute)
            else:
                command = '(foreach map [[px] -> [pxcor] of px] \
                            sort patches map [[py] -> [pycor] of py] \
                            sort patches {0} [[px py p ] -> ask patch px py \
                            [set {1} p]])'.format(datalist, attribute)

            self.link.command(command)

        except jpype.JException as ex:
            print(ex.stacktrace())
            raise NetLogoException(str(ex))
```